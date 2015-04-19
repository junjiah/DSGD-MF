# Distributed SGD in Spark

This repository is for homework 7 of the course [*10-605 Machine Learning with Large Datasets*](http://curtis.ml.cmu.edu/w/courses/index.php/Machine_Learning_with_Large_Datasets_10-605_in_Spring_2015), where we need to implement a stochastic gradient descent algorithm for matrix factorization using Spark.

### Running the code

Make sure you've installed latest Spark and then run following commands. The output user factor and movie factor matrices would be written to disk.

```bash
$SPARK_HOME/bin/spark-submit dsgd_mf.py <num_factors> \
        <num_workers> <num_iterations> \
        <beta_value> <lambda_value> \
        <inputV_filepath> <outputW_filepath> <outputH_filepath> 
```

`experiment.sh` and `experiment_data.txt` are scripts and results for the required experiments from the handout.

### Design and implementation

This section briefly highlights some design choices of my implementation. The code itself is quite self-explanatory (hopefully...) so if having any questions please check the code.

#### 1. Matrix statistics

The first step is to collect some statistics from the rating matrix *V*, such as the number of rows and columns, and most importantly, the number of ratings for each row/column which are used for regularization.

This is achieved using Spark's `rdd.aggregate` method, which takes two functions, one for processing the elements in the RDD, another for combining processed results. I used a dictionary `matrix_stats` to record those statistics, and defined two aforementioned functions called `count_stats_seq` and `count_stats_comb`. As a result, following code accomplishes the job:

```python
matrix_stats = ratings.aggregate(matrix_stats, count_stats_seq,
                                 count_stats_comb)
```

This job takes 3 minutes for the 2G Netflix full dataset.

#### 2. Rating matrix transformation

The next step is to transform the rating matrix such that it contains necessary information for future calculations. First I keyed the ratings by the index of its column block: remember if there are *N* workers, then there are *N\*N* blocks in total and the column block index indicates the vertical block position of an entry in the matrix. In addition, I added the regularization terms to each rating, so as a result each entry of the rating matrix became:

    <col_group>, (<u>, <m>, <r>, <N_i>, <N_j>)

Finally I partitioned all the ratings to *N* parts, while in each partition I grouped the ratings by the strata number. Please refer to the report for how this is achieved to avoid spoilers :)

To illustrate, the code handling such transformation looks like following:

```python
ratings = ratings.map(lambda r: ((r[1] - 1) / blk_col_size,
                                 # value is a 5-element tuple
                                 (r[0], r[1], r[2],
                                  rating_per_user_b.value[r[0]],
                                  rating_per_movie_b.value[r[1]]))) \
                 .partitionBy(num_workers) \
                 .mapPartitionsWithIndex(pack_by_strata,
                                         preservesPartitioning=True) \
                 .cache()
```

where `pack_by_strata` is the function to group ratings by its strata, which is determined by the row and column block indexes.

#### 3. Factor matrices

The user factor matrix ***W*** and movie factor matrix ***H*** are constructed as a NumPy 2-dimensional array and then broadcasted to workers. After filtering out rating entries which don't belong to current strata, I calculate the gradients, update the broadcasted factor matrices at workers then collect them to update the the actual factor matrices in the driver. 

As you can imagine, during each iteration I have to broadcast the factor matrices to workers, so it's not memory efficient.

The corresponding code is as following:

```python
updated = ratings \
    .filter(lambda s: s[0][0] == main_iter % num_workers) \
    .map(update, preservesPartitioning=True) \
    .collect()
```

The second line chooses ratings of current strata, and `update` function does the hard job to calculate the updated factor matrices.

For your reference, on the full dataset my implementation will need about 1.6 minutes for each iteration.

### Others

There are other branches inside this repo, namely:

- **save-memory**: this is for Autolab evaluation. I added some tricks like calling `gc.collect()` or user 16 bit float data type matrices to reduce the memory usage. But it doesn't work very well (scoring 8.6).
- **separate-rdd**: I tried to store  ***W*** and ***H*** also as RDDs, but somehow they didn't provide much benefit.
















