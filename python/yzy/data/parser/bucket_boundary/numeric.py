import pandas
import tensorflow as tf
import argparse as ap
import random
import numpy as np
import gc
import sys

parser = ap.ArgumentParser(description="csv2tfrecord")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ap.ArgumentTypeError('Boolean value expected.')


parser.add_argument("--input_csv", type=str, default=None)
parser.add_argument("--is_input_test", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_input_test")

parser.add_argument("--is_train_to_train_test", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_train_to_train_test")
parser.add_argument("--train_samples_percent", type=float, default=None)
parser.add_argument("--output_train_tfrecord", type=str, default=None)
parser.add_argument("--output_test_tfrecord", type=str, default=None)

parser.add_argument("--is_test_to_test", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_test_to_test")
parser.add_argument("--is_train_to_train", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_train_to_train")
parser.add_argument("--is_statics_bucket", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_statics_bucket")
parser.add_argument("--statics_bucket_num", type=int, default=None)

parser.add_argument("--is_output_id_file", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_output_id_file")
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)

FLAGS, leftovers = parser.parse_known_args()

usecols = list(range(13, 33, 1))

import glob

input_file_list = glob.glob("/data/yezhengyuan/liveme_yezhengyuan_model500/tensorflow_training/tmp/training_samples/shortVideoBase/basicDataAll/20180501" + "/*.gz")

dataframe = pandas.concat([pandas.read_csv(input_file, header=None, usecols=usecols, delimiter="\t", compression="gzip") for input_file in input_file_list])

column_names = list(dataframe.columns.values)
print(column_names)
print(type(column_names))
print(type(dataframe))

df_memory_usage = dataframe.memory_usage(index=True, deep=True)
print(df_memory_usage)
df_memory_usage_sum = dataframe.memory_usage(index=True, deep=True).sum()
print(df_memory_usage_sum)

gc.collect()

if FLAGS.is_train_to_train_test:
    pass

elif FLAGS.is_statics_bucket:

    def chunks_boundary(l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i]


    for elem in usecols:
        # if elem["format"]["csv"]["type"] in list(["int64", "float"]):
        value_list = []
        bucket_column_name = elem
        for value in dataframe[bucket_column_name]:
            try:
                if str(float(value)) == 'nan':
                    # print(value)
                    continue
                else:
                    value_list.append(float(value))
            except:
                continue
        sorted_list = sorted(value_list)

        sorted_list_len = len(sorted_list)
        bucket_list = [i for i in chunks_boundary(sorted_list, int(sorted_list_len / FLAGS.statics_bucket_num))]
        print(bucket_column_name)
        print(sorted_list_len)
        distinct_list = list(set(bucket_list))
        distinct_list.sort()
        print(distinct_list)

else:
    pass
