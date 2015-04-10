import os
from scipy import sparse
import sys
import numpy as np
from numpy.random import rand
from numpy import matrix

from operator import itemgetter
from pyspark import SparkContext, SparkConf


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

    sc = SparkContext('local', 'Distributed Stochastic Gradient Descent')

    if os.path.isdir(inputV_filepath):
        rating_files = sc.wholeTextFiles(inputV_filepath)
        ratings = rating_files.flatMap(
            lambda pair: get_rating(pair[1])).cache()
    else:
        ratings = sc.textFile(inputV_filepath).map(
            lambda line: map(int, line.split(','))).cache()

    # row / col sizes are max user / movie index
    # TODO: DEBUG
    row_number = ratings.max(key=itemgetter(0))[0] + 1
    col_number = ratings.max(key=itemgetter(1))[1] + 1

    # build W and H
    user_factor = matrix(rand(row_number, num_factors))
    movie_factor = matrix(rand(col_number, num_factors))

    # slice ratings matrix to column groups
    # TODO: what if not exact division
    block_column_size = col_number / num_workers
    block_row_size = row_number / num_workers

    strata = range(num_workers)

    def in_strata(rating_entry):
        user, movie, _ = rating_entry
        # TODO
        column_group = (movie - 0) / block_column_size
        return (strata[column_group] * block_row_size <= (user - 0) <
                (strata[column_group] + 1) * block_row_size)

    def update(group, u_f_p, m_f_p):
        """

        :param group:
        :param u_f_p:
        :param m_f_p:
        :return:
        """
        column_group, rating_entries = group
        i = 0

        for user, movie, rating in rating_entries:
            # transform real indexes to partitioned factor matrix indexes
            # TODO
            user_index = (user - 0) % block_row_size
            movie_index = (movie - 0) % block_column_size

            pred_rating = u_f_p[user_index, :] * \
                          m_f_p[movie_index, :].T

            if i % 500 == 0:
                print 'pred: ' + str(pred_rating)

            i += 1

            # TODO: add regularization terms
            # u_f_p[user_index, :] -= beta_value * (
            #     -2 * (rating - pred_rating) * m_f_p[movie_index, :])
            u_gradient = -2 * (rating - pred_rating) * m_f_p[movie_index, :] + \
                2 * lambda_value / col_number * u_f_p[user_index, :]
            u_f_p[user_index, :] -= beta_value * u_gradient

            m_gradient = -2 * (rating - pred_rating) * u_f_p[user_index, :] + \
                2 * lambda_value / row_number * m_f_p[movie_index, :]
            m_f_p[movie_index, :] -= beta_value * m_gradient

        return column_group, (u_f_p, m_f_p)

    for _ in range(num_iterations):
        # split factor matrices
        user_factors = np.split(user_factor, num_workers, axis=0)
        movie_factors = np.split(movie_factor, num_workers, axis=0)

        updated = ratings \
            .filter(lambda r: in_strata(r)) \
            .groupBy(lambda r: r[1] / block_column_size, num_workers) \
            .map(lambda group: update(group,
                                      user_factors[strata[group[0]]],
                                      movie_factors[group[0]])) \
            .collect()

        # aggregate the updates
        for column_group, (updated_u_f, updated_m_f) in updated:
            update_start = strata[column_group] * block_row_size
            update_end = (strata[column_group] + 1) * block_row_size
            user_factor[update_start:update_end, :] = updated_u_f

            update_start = column_group * block_column_size
            update_end = (column_group + 1) * block_column_size
            movie_factor[update_start:update_end, :] = updated_m_f

        # shift the strata
        strata.append(strata.pop(0))

    rating_matrix = sparse.dok_matrix((col_number, row_number), dtype=float)
    for u, m, r in ratings.collect():
        rating_matrix[(m, u)] = r

    pred_ratings = movie_factor * user_factor.T

    diff = rating_matrix.tocsr() - pred_ratings
    rmse = np.sqrt(np.sum(np.power(diff, 2)) / col_number * row_number)
    print rmse

    sc.stop()