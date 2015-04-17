import collections
import os
import sys
import numpy as np

from numpy.random import rand
from pyspark import SparkContext
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

    sc = SparkContext(appName='Distributed Stochastic Gradient Descent')

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
    matrix_stats = {
        'row_num': 0,
        'col_num': 0,
        'rating_per_user': collections.defaultdict(int),
        'rating_per_movie': collections.defaultdict(int)
    }

    def count_stats_seq(stat, rating_entry):
        # this is an rating entry
        u, m, _ = rating_entry
        if u > stat['row_num']:
            stat['row_num'] = u
        if m > stat['col_num']:
            stat['col_num'] = m

        stat['rating_per_user'][u] += 1
        stat['rating_per_movie'][m] += 1
        return stat

    def count_stats_comb(stat1, stat2):
        u, m = stat2['row_num'], stat2['col_num']
        if u > stat1['row_num']:
            stat1['row_num'] = u
        if m > stat1['col_num']:
            stat1['col_num'] = m

        u_count, m_count = stat1['rating_per_user'], stat1['rating_per_movie']
        for key, val in stat2['rating_per_user'].iteritems():
            u_count[key] += val
        for key, val in stat2['rating_per_movie'].iteritems():
            m_count[key] += val

        return stat1

    matrix_stats = ratings.aggregate(matrix_stats, count_stats_seq,
                                     count_stats_comb)

    row_num, col_num = matrix_stats['row_num'], matrix_stats['col_num']
    rating_per_user = matrix_stats['rating_per_user']
    rating_per_movie = matrix_stats['rating_per_movie']

    print 'collected corpus statistics, row num: %d, col num: %d' % (row_num,
                                                                     col_num)

    # build W and H
    u_factor = rand(row_num, num_factors)
    m_factor = rand(col_num, num_factors)
    # change the data type to save memory
    u_factor = u_factor.astype(np.float32, copy=False)
    m_factor = m_factor.astype(np.float32, copy=False)

    # determine block size
    blk_col_size = (col_num - 1) / num_workers + 1
    blk_row_size = (row_num - 1) / num_workers + 1

    def pack_by_strata(col_group, partition_iter):
        strata = collections.defaultdict(list)
        perm = range(num_workers)
        for _ in range(col_group):
            perm.insert(0, perm.pop())

        for entry in partition_iter:
            _, (u, m, _, _, _) = entry
            row_group = (u - 1) / blk_row_size
            strata[(perm[row_group], row_group, col_group)].append(entry[1])

        for item in strata.items():
            yield item

    # add N_i, N_j for each rating entry
    rating_per_user_b = sc.broadcast(rating_per_user)
    rating_per_movie_b = sc.broadcast(rating_per_movie)
    # step1: map to :(<col-group>, (<u> <m> <r> <N_i> <N_j>))
    ratings = ratings.map(lambda r: ((r[1] - 1) / blk_col_size,
                                     # value is a 5-element tuple
                                     (r[0], r[1], r[2],
                                      rating_per_user_b.value[r[0]],
                                      rating_per_movie_b.value[r[1]]))) \
                     .partitionBy(num_workers) \
                     .mapPartitionsWithIndex(pack_by_strata,
                                             preservesPartitioning=True) \
                     .cache()

    def calculate_loss(pred_rating, true_rating):
        error, n = 0.0, 0
        for _, entries in true_rating:
            for u, m, r, _, _ in entries:
                error += (r - pred_rating[u - 1, m - 1]) ** 2
                n += 1

        print 'loss: %f, RMSE: %f' % (error,
                                      np.sqrt(error / n))

    def update(block):
        # only one entry in each partition,
        # corresponding to this specific strata
        (_, row_group, col_group), entries = block

        row_start = row_group * blk_row_size
        col_start = col_group * blk_col_size
        u_f_p = u_factor_b.value[row_start:row_start + blk_row_size, :]
        m_f_p = m_factor_b.value[col_start:col_start + blk_col_size, :]

        num_updated = 0
        # value of data is the rating entry
        for u, m, r, n_u, n_m in entries:
            # num_prev_update is retrieved automatically
            learning_rate = pow(TAO_0 + num_prev_update + num_updated,
                                -beta_value)

            # transform real indexes to partitioned factor matrix indexes
            user_index = (u - 1) % blk_row_size
            movie_index = (m - 1) % blk_col_size

            rating_diff = r - np.dot(
                u_f_p[user_index, :], m_f_p[movie_index, :])

            u_gradient = -2 * rating_diff * m_f_p[movie_index, :] + \
                2 * lambda_value / n_u * u_f_p[user_index, :]
            u_f_p[user_index, :] -= learning_rate * u_gradient

            m_gradient = -2 * rating_diff * u_f_p[user_index, :] + \
                2 * lambda_value / n_m * m_f_p[movie_index, :]
            m_f_p[movie_index, :] -= learning_rate * m_gradient

            num_updated += 1

        return row_group, col_group, u_f_p, m_f_p, num_updated

    # running DSGD!
    num_prev_update = 0
    for main_iter in range(num_iterations):
        # broadcast factor matrices
        u_factor_b = sc.broadcast(u_factor)
        m_factor_b = sc.broadcast(m_factor)

        updated = ratings \
            .filter(lambda s: s[0][0] == main_iter % num_workers) \
            .map(update, preservesPartitioning=True) \
            .collect()

        u_factor_b.unpersist()
        m_factor_b.unpersist()
        # aggregate the updates
        for row_group, col_group, updated_u, updated_m, iter_num in updated:
            update_start = row_group * blk_row_size
            update_end = (row_group + 1) * blk_row_size
            u_factor[update_start:update_end, :] = updated_u

            update_start = col_group * blk_col_size
            update_end = (col_group + 1) * blk_col_size
            m_factor[update_start:update_end, :] = updated_m

            num_prev_update += iter_num

        # EXPERIMENTS: output evaluation results
        # print 'iteration: %d' % main_iter
        # calculate_loss(np.dot(u_factor, m_factor.T), ratings.collect())

    # simple evaluation
    print 'time usage: %s seconds' % (time.time() - start_time)
    calculate_loss(np.dot(u_factor, m_factor.T), ratings.collect())

    sc.stop()
    # write parameters
    with open(outputW_filepath, 'wb') as f:
        for row in u_factor:
                f.write(','.join(map(str, row)) + '\n')

    with open(outputH_filepath, 'wb') as f:
        for row in m_factor.T:
            f.write(','.join(map(str, row)) + '\n')
