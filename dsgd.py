import collections
import os
import sys
import numpy as np

from numpy.random import rand
from pyspark import SparkContext, SparkConf
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


def extract_stats(stats, rating_entry):
    pass


if __name__ == '__main__':
    # read command line arguments
    num_factors, num_workers, num_iterations = map(int, sys.argv[1:4])
    beta_value, lambda_value = map(float, sys.argv[4:6])
    inputV_filepath, outputW_filepath, outputH_filepath = sys.argv[6:]

    TAO_0 = 100

    conf = SparkConf()
    conf.set('spark.executor.memory', '20g')
    conf.set('spark.driver.memory', '20g')

    sc = SparkContext('local[8]', 'Distributed Stochastic Gradient Descent',
                      conf=conf)

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
            # this is an movie entry
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

    # extract info for strata/blocks
    blk_col_size = (col_num - 1) / num_workers + 1
    blk_row_size = (row_num - 1) / num_workers + 1

    # # add N_i, N_j for each rating entry
    rating_per_user_b = sc.broadcast(rating_per_user)
    rating_per_movie_b = sc.broadcast(rating_per_movie)

    # map to :(<col-group>, (<u> <m> <r> <N_i> <N_j>))
    ratings = ratings.map(lambda r: ((r[1] - 1) / blk_col_size,
                                     # value is a 5-element tuple
                                     (r[0], r[1], r[2],
                                      rating_per_user_b.value[r[0]],
                                      rating_per_movie_b.value[r[1]]))) \
        .cache()

    # build W and H, keyed on their groups
    W, H = rand(row_num, num_factors), rand(col_num, num_factors)
    u_factor, m_factor = [], []
    for i in range(num_workers):
        u_factor.append(W[i * blk_row_size:(i + 1) * blk_row_size, :])
    for i in range(num_workers):
        m_factor.append(H[i * blk_col_size:(i + 1) * blk_col_size, :])
    # clear W, H to prevent later references
    W, H = None, None

    u_factor = sc.parallelize(enumerate(u_factor))
    m_factor = sc.parallelize(enumerate(m_factor))

    strata = range(num_workers)
    rev_strata_map = range(num_workers)

    strata = range(num_workers)

    def in_strata(rating_entry):
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

    def update(val):
        rating_entries, u_f_p, m_f_p = val
        u_f_p = u_f_p.data[0]
        m_f_p = m_f_p.data[0]
        # ratings could be None
        if rating_entries.data:
            for user, movie, rating, num_rated_m, num_rated_u in rating_entries:
                # transform real indexes to partitioned factor matrix indexes
                user_index = (user - 1) % blk_row_size
                movie_index = (movie - 1) % blk_col_size

                pred_rating = np.dot(u_f_p[user_index, :], m_f_p[movie_index, :])

                u_gradient = -2 * (rating - pred_rating) * m_f_p[movie_index, :] + \
                            2 * lambda_value / num_rated_m * \
                            u_f_p[user_index, :]
                u_f_p[user_index, :] -= beta_value * u_gradient

                m_gradient = -2 * (rating - pred_rating) * u_f_p[user_index, :] + \
                            2 * lambda_value / num_rated_u * \
                            m_f_p[movie_index, :]
                m_f_p[movie_index, :] -= beta_value * m_gradient

        return u_f_p, m_f_p

    for i in range(num_iterations):
        updated = ratings \
            .filter(lambda r: in_strata(r)) \
            .groupWith(u_factor.map(lambda f: (rev_strata_map[f[0]], f[1])),
                       m_factor) \
            .mapValues(update) \
            .collect()

        # aggregate the updates
        u_factor, m_factor = [-1] * num_workers, [-1] *num_workers
        for column_group, (updated_u_f, updated_m_f) in updated:
            u_factor[strata[column_group]] = updated_u_f
            m_factor[column_group] = updated_m_f

        # output evaluation results
        print
        print 'iteration: %d' % i
        calculate_loss(np.dot(np.vstack(u_factor), np.vstack(m_factor).T),
                       ratings.collect())

        if i != num_iterations - 1:
            u_factor = sc.parallelize(enumerate(u_factor))
            m_factor = sc.parallelize(enumerate(m_factor))

            # n += block_i

        # shift the strata
        strata.append(strata.pop(0))
        rev_strata_map.insert(0, rev_strata_map.pop())

    # do simple evaluation
    print
    print
    print 'time usage: %s seconds' % (time.time() - start_time)
    calculate_loss(np.dot(np.vstack(u_factor), np.vstack(m_factor).T),
                   ratings.collect())

    sc.stop()
    # write parameters
    # with open(outputW_filepath, 'wb') as f:
    #     for row in u_factor:
    #         f.write(','.join(map(str, row)) + '\n')
    #
    # with open(outputH_filepath, 'wb') as f:
    #     for row in m_factor.T:
    #         f.write(','.join(map(str, row)) + '\n')
