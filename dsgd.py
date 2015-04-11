import os
import sys
import numpy as np

from numpy.random import rand
from operator import itemgetter
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

    sc = SparkContext('local[8]', 'Distributed Stochastic Gradient Descent')

    if os.path.isdir(inputV_filepath):
        rating_files = sc.wholeTextFiles(inputV_filepath)
        ratings = rating_files.flatMap(
            lambda pair: get_rating(pair[1])).cache()
    else:
        ratings = sc.textFile(inputV_filepath).map(
            lambda line: map(int, line.split(','))).cache()

    # measure the starting time
    start_time = time.time()

    # row / col sizes are max user / movie index
    row_num = ratings.max(key=itemgetter(0))[0]
    col_num = ratings.max(key=itemgetter(1))[1]

    # build W and H
    u_factor = rand(row_num, num_factors)
    m_factor = rand(col_num, num_factors)

    # extract info for strata/blocks
    blk_col_size = (col_num - 1) / num_workers + 1
    blk_row_size = (row_num - 1) / num_workers + 1

    strata = range(num_workers)

    # calculate regularization counts
    # TODO: more efficient ways?
    rating_per_row = dict(ratings.groupBy(itemgetter(0)) \
                                 .mapValues(len) \
                                 .collect())
    rating_per_col = dict(ratings.groupBy(itemgetter(1)) \
                                 .mapValues(len) \
                                 .collect())

    def in_strata(rating_entry):
        user, movie, _ = rating_entry
        col_group = (movie - 1) / blk_col_size
        return (strata[col_group] * blk_row_size <= (user - 1) <
                (strata[col_group] + 1) * blk_row_size)

    def update(group, row_start, col_start):
        """
        Update the incoming user/movie factor matrix with given rating entries
        :param group: column group number and rating entries
        :param row_start: starting index of this user partition
        :param col_start: starting index of this movie partition
        :return: a tuple of user factor matrix and movie factor matrix, using
                 the column number as key
        """
        col_group, rating_entries = group

        # partition of user/movie factor matrix
        u_f_p = u_factor[row_start:row_start + blk_row_size, :]
        m_f_p = m_factor[col_start:col_start + blk_col_size, :]

        for user, movie, rating in rating_entries:
            # transform real indexes to partitioned factor matrix indexes
            user_index = (user - 1) % blk_row_size
            movie_index = (movie - 1) % blk_col_size

            pred_rating = np.dot(u_f_p[user_index, :], m_f_p[movie_index, :])

            u_gradient = -2 * (rating - pred_rating) * m_f_p[movie_index, :] + \
                        2 * lambda_value / rating_per_row[user] * \
                        u_f_p[user_index, :]
            u_f_p[user_index, :] -= beta_value * u_gradient

            m_gradient = -2 * (rating - pred_rating) * u_f_p[user_index, :] + \
                        2 * lambda_value / rating_per_col[movie] * \
                        m_f_p[movie_index, :]
            m_f_p[movie_index, :] -= beta_value * m_gradient

        return col_group, u_f_p, m_f_p

    for _ in range(num_iterations):
        # note in map, `preservesPartitioning` is True
        updated = ratings \
            .filter(lambda r: in_strata(r)) \
            .groupBy(lambda r: (r[1] - 1) / blk_col_size, num_workers) \
            .map(lambda group: update(group,
                                      strata[group[0]] * blk_row_size,
                                      group[0] * blk_col_size), True) \
            .collect()

        # aggregate the updates
        for column_group, updated_u_f, updated_m_f in updated:
            update_start = strata[column_group] * blk_row_size
            update_end = (strata[column_group] + 1) * blk_row_size
            u_factor[update_start:update_end, :] = updated_u_f

            update_start = column_group * blk_col_size
            update_end = (column_group + 1) * blk_col_size
            m_factor[update_start:update_end, :] = updated_m_f

        # shift the strata
        strata.append(strata.pop(0))

    # do simple evaluation
    end_time = time.time()
    pred_ratings = np.dot(u_factor, m_factor.T)
    rmse, rating_count = 0.0, 0
    for u, m, r in ratings.collect():
        rmse += (r - pred_ratings[u - 1, m - 1]) ** 2
        rating_count += 1

    sc.stop()
    loss = rmse
    rmse = np.sqrt(rmse / rating_count)
    print
    print
    print 'RMSE: ' + str(rmse)
    print 'loss: ' + str(loss)
    print 'time usage: %s seconds' % (end_time - start_time)

    # write parameters
    with open(outputW_filepath, 'wb') as f:
        for row in u_factor:
            f.write(','.join(map(str, row)) + '\n')

    with open(outputH_filepath, 'wb') as f:
        for row in m_factor.T:
            f.write(','.join(map(str, row)) + '\n')