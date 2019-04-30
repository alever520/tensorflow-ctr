"""cal auc"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import shutil
import sys
from itertools import groupby
import gzip
import sys
import json
from multiprocessing import Process,Lock,Manager
import itertools
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()

# parser.add_argument(
#     '--dwell_label_thresh', type=int, default=1, help='dwell_label_thresh')
#
# parser.add_argument(
#     '--share_label_thresh', type=int, default=1, help='share_label_thresh')
#
# parser.add_argument(
#     '--like_label_thresh', type=int, default=1, help='like_label_thresh')
#
# parser.add_argument(
#     '--follow_label_thresh', type=int, default=1, help='follow_label_thresh')

parser.add_argument(
    '--input_feature_data', type=str, default='/tmp/input_feature_data.data',
    help='Path to the input_feature_data.')

parser.add_argument(
    '--input_score_data', type=str, default='/tmp/input_score_data.data',
    help='Path to the input_score_data.')

parser.add_argument(
    '--time_tag', type=str, default='2018010100',
    help='time_tag for output')

parser.add_argument(
    '--project', type=str, default='model_name',
    help='project name')

parser.add_argument(
    '--auc_multi_label_json', type=str, default='/tmp/auc_multi_label_json.json',
    help='Path to the model training json.')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument("--is_input_gzip", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_input_gzip")

parser.add_argument("--is_input_tfrecord", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_input_tfrecord")

parser.add_argument("--is_input_feature_score_merged", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_input_feature_score_merged")

parser.add_argument("--is_cal_rmse", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_cal_rmse")


def read_input_one_file(features_path):
    if FLAGS.is_input_gzip:
        features = [line.rstrip('\n') for line in gzip.open(features_path)]
    else:
        features = [line.rstrip('\n') for line in open(features_path)]
    return map(translate_one_file, features)


def translate_one_file(record):
    items = record.split('\t')
    # label = int(items[3])
    label_dict = {k: float(int(items[v["idx_in_csv"]]) >= v["thresh"]) for k, v in label_info.items()}
    # label_dict = dict(map(lambda kv: (kv[0], f(kv[1])), label_info.iteritems()))
    #score = round(sum([float(l) * r for l, r in zip(predict.split('\t'), score_weight)]), 6)
    score = [float(items[score_idx]) for score_idx in auc_multi_label_json_dict["score_idxs"]]
    sample_info = []
    for idx in csv_column_idx_needs:
        sample_info.append(items[idx])
    return sample_info, label_dict, score


def translate((feature, predict)):
    items = feature.split('\t')
    # label = int(items[3])
    label_dict = {k: float(int(items[v["idx_in_csv"]]) >= v["thresh"]) for k, v in label_info.items()}
    # label_dict = dict(map(lambda kv: (kv[0], f(kv[1])), label_info.iteritems()))
    #score = round(sum([float(l) * r for l, r in zip(predict.split('\t'), score_weight)]), 6)
    score = [float(l) for l in predict.split('\t')]
    sample_info = []
    for idx in csv_column_idx_needs:
        sample_info.append(items[idx])
    # sample_info = [items[user_id_idx_in_csv], items[_POS_INDEX['u_country']]]
    return sample_info, label_dict, score


def read_tfrecord_as_features(features_path,predicts_path):
    predicts = [line.rstrip('\n') for line in open(predicts_path)]
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    example = tf.train.Example()
    res = []
    for (record, predict) in zip(tf.python_io.tf_record_iterator(features_path, options=options), predicts):
        score = [float(l) for l in predict.split('\t')]
        example.ParseFromString(record)
        label_dict = label_info.copy()
        if FLAGS.is_cal_rmse:
            for k, v in label_dict.items():
                label_dict[k] = float(example.features.feature[k].float_list.value[0])
            # label_dict = {k: float(example.features.feature[k].float_list.value[0]) for k, v in label_info.items()}
        else:
            for k, v in label_dict.items():
                label_dict[k] = float(example.features.feature[k].float_list.value[0] >= v["thresh"])
            # label_dict = {k: float(example.features.feature[k].float_list.value[0] >= v["thresh"]) for k, v in label_info.items()}
        sample_info = [example.features.feature[k].bytes_list.value[0] for k in tfrecord_bytes_key_needs]
        res.append((sample_info, label_dict, score))
    return res


def read_input(features_path, predicts_path):
    if FLAGS.is_input_gzip:
        features = [line.rstrip('\n') for line in gzip.open(features_path)]
    else:
        features = [line.rstrip('\n') for line in open(features_path)]
    predicts = [line.rstrip('\n') for line in open(predicts_path)]
    return map(translate, zip(features, predicts))


def formatdata(data, thr):
    datalist = dict()
    for uid, label, score in data:
        if score not in datalist:
            datalist[score] = [0, 0]
        if label >= thr:
            datalist[score][0] += 1
        else:
            datalist[score][1] += 1
    data = [[k, v[0], v[1]] for k, v in datalist.iteritems()]
    return sorted(data, key=lambda x: -x[0])


def format_data_for_auc(data):
    datalist = dict()
    for uid, label, score in data:
        if score not in datalist:
            datalist[score] = [0, 0]
        if label >= 1.0:
            datalist[score][0] += 1
        else:
            datalist[score][1] += 1
    data = [[k, v[0], v[1]] for k, v in datalist.iteritems()]
    return sorted(data, key=lambda x: -x[0])


def auc_origin(datalist):
    totalP = sum([x[1] for x in datalist])
    totalN = sum([x[2] for x in datalist])
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    sumP = 0
    sumN = 0
    sumArea = 0
    for e in datalist:
        sumP += e[1]
        sumN += e[2]
        x1, y1 = x2, y2
        if totalN != 0 and totalP != 0:
            x2 = 1.0 * sumN / totalN
            y2 = 1.0 * sumP / totalP
        if e[2] != 0:
            sumArea += (x2 - x1) * (y2 + y1) / 2
    return sumArea


def auc_speedup(datalist):
    totalP = 0
    totalN = 0
    for x in datalist:
        totalP += x[1]
        totalN += x[2]
    if totalN == 0 or totalP == 0:
        return 0
    reTotalP = 1.0 / totalP
    reTotalN = 1.0 / totalN
    reTotalPN = reTotalP * reTotalN * 0.5

    sumP = 0
    sumN = 0
    sumArea = 0
    for e in datalist:
        x1 = sumN
        y1 = sumP
        sumP += e[1]
        sumN += e[2]
        if e[2] != 0:
            sumArea += (sumN - x1) * (sumP + y1)

    return sumArea * reTotalPN


def per_auc(sorted_data):
    #sorted_data = sorted(data, key=lambda i: i[0])

    res = dict()

    for key in label_info.keys():

        data_uid_label_score = [[uid, label_dict[key], score] for uid, label_dict, score in sorted_data]

        if FLAGS.is_cal_rmse:
            user_metrics_sum = 0
            users = 0
            user_metrics_weighted_sum = 0
            samples = 0
            all_sample_se = 0

            # sorted outside this function already
            for uid, group in groupby(data_uid_label_score, lambda i: i[0]):
                group = list(group)
                user_samples = len(group)

                single_user_sample_se = sum((i[1] - i[2])**2 for i in group)
                single_user_rmse = (single_user_sample_se / user_samples)**0.5

                samples += user_samples
                user_metrics_weighted_sum += user_samples * single_user_rmse
                users += 1
                user_metrics_sum += single_user_rmse
                all_sample_se += single_user_sample_se
            all_sample_rmse = 0.0 if samples == 0 else (all_sample_se / samples) ** 0.5
            weight_user_auc = 0.0 if samples == 0 else user_metrics_weighted_sum / samples
            per_user_auc = 0.0 if users == 0 else user_metrics_sum / users
            res[key] = [all_sample_rmse, weight_user_auc, per_user_auc]
        else:
            user_auc_sum = 0
            users = 0
            user_auc_weighted_sum = 0
            samples = 0
            #sorted outside this function already
            for uid, group in groupby(data_uid_label_score, lambda i: i[0]):
                group = list(group)
                user_samples = len(group)
                if all(i[1] < 1.0 for i in group):
                    continue
                if all(i[1] >= 1.0 for i in group):
                    continue
                user_format_data = format_data_for_auc(group)
                user_auc = auc_speedup(user_format_data)
                samples += user_samples
                user_auc_weighted_sum += user_samples * user_auc
                users += 1
                user_auc_sum += user_auc
            format_sort_data = format_data_for_auc(data_uid_label_score)
            all_auc = auc_speedup(format_sort_data)
            weight_user_auc = 0.0 if samples == 0 else user_auc_weighted_sum / samples
            per_user_auc = 0.0 if users == 0 else user_auc_sum / users
            res[key] = [all_auc, weight_user_auc, per_user_auc]

    return res


def parallel_auc(score_weight_pair, pre_data, lock, res_dict):
    aucs = []
    conf_name, score_weight = score_weight_pair
    data = [(i[0], i[1], round(sum([l * r for l, r in zip(i[2], score_weight)]), 6)) for i in pre_data]

    for elem in processed_user_groups:
        name = ""
        for pos, val in elem:
            name += str(val) + "\t"
        name = name.rstrip("\t")
        data_dict[name] = list()
        for single_data in data:
            if all(val == "ALL" or single_data[0][pos] == val for pos, val in elem):
                data_dict[name].append((single_data[0][0], single_data[1], single_data[2]))

    # data_dict['ALL'] = [(i[0][0], i[1], i[2]) for i in data]
    # for j, conf in enumerate(FIELD_CONF):
    #     for k, v in conf.iteritems():
    #         data_dict[k] = [(i[0][0], i[1], i[2]) for i in data if i[0][j + 1] == v]
    for k, data_cate in data_dict.iteritems():
        # print("\t".join(["LogLiveme", k, str(len(data_cate))]))
        # over_all_auc, p_auc1, p_auc2 = per_auc(data_cate)
        for auc_type, auc_list in per_auc(data_cate).items():
            sample_auc = auc_list[0]
            weighted_user_auc = auc_list[1]
            user_auc = auc_list[2]
            aucs.append('%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f' % (
                FLAGS.project + conf_name, FLAGS.time_tag, k, auc_type, sample_auc, weighted_user_auc, user_auc))
    lock.acquire()
    res_dict[conf_name] = aucs
    lock.release()


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    auc_multi_label_json_dict = json.load(open(FLAGS.auc_multi_label_json))

    #user_id_idx_in_sample_info = auc_multi_label_json_dict["user_id"]["idx_in_sample_info"]
    score_weight_dict = auc_multi_label_json_dict["score_weight"]
    label_info = auc_multi_label_json_dict["label_info"]
    user_groups = auc_multi_label_json_dict["user_groups"]

    listoflists = [[(elem["idx_in_sample_info"], subelem) for subelem in elem["in"]] for elem in user_groups]
    processed_user_groups = list(itertools.product(*listoflists))

    # Clean up the model directory if present
    # shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    if FLAGS.is_input_tfrecord:
        tfrecord_bytes_key_needs = auc_multi_label_json_dict["tfrecord_bytes_key_needs"]
        #tfrecord_float_key_needs = auc_multi_label_json_dict["tfrecord_float_key_needs"]
        # print(time.time(), "before read_tfrecord_as_features")
        full_data = read_tfrecord_as_features(FLAGS.input_feature_data, FLAGS.input_score_data)
        # print(time.time(), "after read_tfrecord_as_features")
    else:
        csv_column_idx_needs = auc_multi_label_json_dict["csv_column_idx_needs"]
        if FLAGS.is_input_feature_score_merged:
            full_data = read_input_one_file(FLAGS.input_feature_data)
        else:
            full_data = read_input(FLAGS.input_feature_data, FLAGS.input_score_data)
    data_dict = {}
    # reg_data = [i for i in full_data if i[0][2] == '1']
    # vis_data = [i for i in full_data if i[0][2] == '0']
    # ava_data = [k for k in [reg_data, vis_data] if k]
    # if not ava_data:
    ava_data = [full_data]

    aucs = list()
    for pre_data in ava_data:
        pre_data = sorted(pre_data, key=lambda elem: elem[0][0])

        jobs = list()
        res_dict = Manager().dict()
        lock = Lock()

        for elem in score_weight_dict.iteritems():
            job = Process(target=parallel_auc, args=(elem, pre_data, lock, res_dict))
            jobs.append(job)
            job.start()

        for job in jobs:
            job.join()

        aucs.extend([item for k, v in res_dict.items() for item in v])

    print('\n'.join(aucs))
