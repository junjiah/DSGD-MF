#!/bin/bash

for i in {2,3,4,5,6,7,8,9,10}
do
    ~/Workspace/10605/spark-1.3.0-bin-hadoop2.4/bin/spark-submit --master local[$i] --executor-memory 2G --driver-memory 10G ~/Workspace/10605/hw7/DSGD/dsgd.py 20 $i 30 0.6 0.1 /Users/hejunjia1911/Workspace/10605/hw7/nf_subsample.csv w.csv h.csv > $i.txt
done

for (( i = 10; i <= 100; i += 10 ))
do
    ~/Workspace/10605/spark-1.3.0-bin-hadoop2.4/bin/spark-submit --master local[10] --executor-memory 2G --driver-memory 10G ~/Workspace/10605/hw7/DSGD/dsgd.py $i 10 30 0.6 0.1 /Users/hejunjia1911/Workspace/10605/hw7/nf_subsample.csv w.csv h.csv > $i.txt
done

for (( i = 5; i <= 9; i += 1 ))
do
    beta=0`echo "scale=1; $i/10" | bc -l`
    ~/Workspace/10605/spark-1.3.0-bin-hadoop2.4/bin/spark-submit --master local[10] --executor-memory 2G --driver-memory 10G ~/Workspace/10605/hw7/DSGD/dsgd.py 20 10 30 $beta 0.1 /Users/hejunjia1911/Workspace/10605/hw7/nf_subsample.csv w.csv h.csv > $beta.txt
done
