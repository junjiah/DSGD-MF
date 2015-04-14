import collections
import os
import sys
import numpy as np

from numpy.random import rand
from operator import itemgetter
from pyspark import SparkContext, SparkConf, AccumulatorParam
import time


def get_rating(file_content):
    """
    Transform following format file to triples
        movieId:
        userId1,rating1,date1
        userId2,rating2,date2
    """
    movie_id = -1
    ratings = []
    for line in file_content.split('\n'):
        try:
            if ':' in line:
                movie_id = int(line[:-1])
            else:
                user_id, rating = map(int, line.split(',')[:2])
                ratings.append((user_id, movie_id, rating))
        except:
            pass

    return ratings


if __name__ == '__main__':
    # read command line arguments
    num_factors, num_workers, num_iterations = map(int, sys.argv[1:4])
    beta_value, lambda_value = map(float, sys.argv[4:6])
    inputV_filepath, outputW_filepath, outputH_filepath = sys.argv[6:]

    TAO_0 = 100

    sc = SparkContext('local[8]', 'Distributed Stochastic Gradient Descent')

    if os.path.isfile(inputV_filepath):
        ratings = sc.textFile(inputV_filepath).map(
            lambda line: map(int, line.split(',')))
    else:
        # directory, or on HDFS
        rating_files = sc.wholeTextFiles(inputV_filepath)
        ratings = rating_files.flatMap(
            lambda pair: get_rating(pair[1]))

    # measure the starting time
    start_time = time.time()

    # get corpus statistics
    corpus_stats = {
        'row_num': 0,
        'col_num': 0,
        'rating_per_user': collections.defaultdict(int),
        'rating_per_movie': collections.defaultdict(int)
    }

    def count_stats(v, stat):
        if type(v) == list or type(v) == tuple:
            # this is an rating entry
            u, m, _ = v
            if u > stat['row_num']:
                stat['row_num'] = u
            if m > stat['col_num']:
                stat['col_num'] = m

            stat['rating_per_user'][u] += 1
            stat['rating_per_movie'][m] += 1
        else:
            # another stat
            u, m = v['row_num'], v['col_num']
            if u > stat['row_num']:
                stat['row_num'] = u
            if m > stat['col_num']:
                stat['col_num'] = m

            for key, val in v['rating_per_user'].iteritems():
                stat['rating_per_user'][key] += val
            for key, val in v['rating_per_movie'].iteritems():
                stat['rating_per_movie'][key] += val

        return stat

    corpus_stats = ratings.fold(corpus_stats, count_stats)

    row_num, col_num = corpus_stats['row_num'], corpus_stats['col_num']
    rating_per_user = corpus_stats['rating_per_user']
    rating_per_movie = corpus_stats['rating_per_movie']

    print 'collected corpus statistics, row num: %d, col num: %d' % (row_num,
                                                                     col_num)

    # build W and H
    u_factor = rand(row_num, num_factors)
    m_factor = rand(col_num, num_factors)

    # extract info for strata/blocks
    blk_col_size = (col_num - 1) / num_workers + 1
    blk_row_size = (row_num - 1) / num_workers + 1

    # add N_i, N_j for each rating entry
    rating_per_user_b = sc.broadcast(rating_per_user)
    rating_per_movie_b = sc.broadcast(rating_per_movie)
    # map to :(<col-group>, (<u> <m> <r> <N_i> <N_j>))
    ratings = ratings.map(lambda r: ((r[1] - 1) / blk_col_size,
                                     # value is a 5-element tuple
                                     (r[0], r[1], r[2],
                                      rating_per_user_b.value[r[0]],
                                      rating_per_movie_b.value[r[1]]))) \
        .cache()

    # shipped to workers when necessary, this is small so not necessary
    # to broadcast
    strata = range(num_workers)

    def in_strata(rating_entry):
        """
        A function to keep ratings inside the current strata.
        Note the `strata` is automatically copied.
        """
        col_group = rating_entry[0]
        user = rating_entry[1][0]
        return (strata[col_group] * blk_row_size <= (user - 1) <
                (strata[col_group] + 1) * blk_row_size)

    def calculate_loss(pred_rating, true_rating):
        error = 0.0
        for _, (u, m, r, _, _) in true_rating:
            error += (r - pred_rating[u - 1, m - 1]) ** 2

        print 'loss: %f, RMSE: %f' % (error,
                                      np.sqrt(error / len(true_rating)))

    def update(col_group, partition):
        row_start = strata[col_group] * blk_row_size
        col_start = col_group * blk_col_size

        u_f_p = u_factor_b.value[row_start:row_start + blk_row_size, :]
        m_f_p = m_factor_b.value[col_start:col_start + blk_col_size, :]

        num_updated = 0

        for u, m, r, u_rating_num, m_rating_num in partition:
            # num_prev_update is retrieved automatically
            learning_rate = pow(TAO_0 + num_prev_update + num_updated,
                                -beta_value)

            # transform real indexes to partitioned factor matrix indexes
            user_index = (u - 1) % blk_row_size
            movie_index = (m - 1) % blk_col_size

            rating_diff = r - np.dot(
                u_f_p[user_index, :], m_f_p[movie_index, :])

            u_gradient = -2 * rating_diff * m_f_p[movie_index, :] + \
                2 * lambda_value / u_rating_num * u_f_p[user_index, :]
            u_f_p[user_index, :] -= learning_rate * u_gradient

            m_gradient = -2 * rating_diff * u_f_p[user_index, :] + \
                2 * lambda_value / m_rating_num * m_f_p[movie_index, :]
            m_f_p[movie_index, :] -= learning_rate * m_gradient

            num_updated += 1

        yield col_group, u_f_p, m_f_p, num_updated

    # running DSGD!
    num_prev_update = 0
    for main_iter in range(num_iterations):
        # broadcast factor matrices
        u_factor_b = sc.broadcast(u_factor)
        m_factor_b = sc.broadcast(m_factor)

        updated = ratings \
            .filter(lambda r: in_strata(r)) \
            .partitionBy(num_workers) \
            .mapPartitionsWithIndex(update) \
            .collect()

        # aggregate the updates
        for column_group, updated_u_f, updated_m_f, block_i in updated:
            if column_group == -1:
                continue
            update_start = strata[column_group] * blk_row_size
            update_end = (strata[column_group] + 1) * blk_row_size
            u_factor[update_start:update_end, :] = updated_u_f

            update_start = column_group * blk_col_size
            update_end = (column_group + 1) * blk_col_size
            m_factor[update_start:update_end, :] = updated_m_f

            num_prev_update += block_i

        # shift the strata
        strata.append(strata.pop(0))
        # output evaluation results
        print
        print 'iteration: %d' % main_iter
        # calculate_loss(np.dot(u_factor, m_factor.T), ratings.collect())

    # do simple evaluation
    print
    print
    print 'time usage: %s seconds' % (time.time() - start_time)
    # TODO: try another way to calculate the loss
    calculate_loss(np.dot(u_factor, m_factor.T), ratings.collect())

    sc.stop()
    # write parameters
    # with open(outputW_filepath, 'wb') as f:
    #     for row in u_factor:
    #         f.write(','.join(map(str, row)) + '\n')
    #
    # with open(outputH_filepath, 'wb') as f:
    #     for row in m_factor.T:
    #         f.write(','.join(map(str, row)) + '\n')
