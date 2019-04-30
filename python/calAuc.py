# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
from itertools import groupby
import gzip
import sys

_CSV_COLUMNS = [
    'SVU_TYPE', 'SVV_VID', 'SVU_UID', 'SVA_UID',
    'SVDWELL', 'SVSHARE', 'SVJOIN', 'SVCOMMENT', 'SVEXPOSE', 'SVFOLLOW', 'SVLIKE',
    'SVU_COUNTRY', 'SVU_OS', 'SVV_COUNTRY',

    'SVV_DURATION', 'SVV_DWELL', 'SVV_JOIN', 'SVV_SHARE', 'SVV_CLICK', 'SVV_COMMENT', 'SVV_FOLLOW', 'SVV_LIKE',
    'SVV_CLICKRATE', 'SVV_LIKERATE', 'SVV_COMMENTRATE', 'SVV_SHARERATE', 'SVV_FOLLOWRATE', 'SVV_WATCHALLRATE'
]

_CSV_COLUMN_DEFAULTS = [
    [''], [''], [''], [''],
    [0], [0], [0], [0], [0], [0], [0],
    [''], [''], [''],
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
]

_POS_INDEX = {
    'u_uid': _CSV_COLUMNS.index('SVU_UID'),
    'u_country': _CSV_COLUMNS.index('SVU_COUNTRY'),
}
# first pos, second thresh
label_conf_dict = {'SVDWELL': [_CSV_COLUMNS.index('SVDWELL'), 17],
                   'SVSHARE': [_CSV_COLUMNS.index('SVSHARE'), 1],
                   'SVFOLLOW': [_CSV_COLUMNS.index('SVFOLLOW'), 1],
                   'SVLIKE': [_CSV_COLUMNS.index('SVLIKE'), 1]}

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dwell_label_thresh', type=int, default=1, help='dwell_label_thresh')

parser.add_argument(
    '--share_label_thresh', type=int, default=1, help='share_label_thresh')

parser.add_argument(
    '--like_label_thresh', type=int, default=1, help='like_label_thresh')

parser.add_argument(
    '--follow_label_thresh', type=int, default=1, help='follow_label_thresh')

parser.add_argument(
    '--input_feature_data', type=str, default='/tmp/census_data/adult.data',
    help='Path to the input_feature_data.')

parser.add_argument(
    '--input_score_data', type=str, default='/tmp/census_data/adult.data',
    help='Path to the input_score_data.')

parser.add_argument(
    '--time_tag', type=str, default='2018010100',
    help='time_tag for output')

parser.add_argument(
    '--project', type=str, default='model_name',
    help='project name')


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

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def translate(feature, predict):
    items = feature.split('\t')
    # label = int(items[3])
    label_dict = {k: float(int(items[v[0]]) >= v[1]) for k, v in label_conf_dict.items()}
    # label_dict = dict(map(lambda kv: (kv[0], f(kv[1])), label_conf_dict.iteritems()))
    score = round(float(predict), 6)
    sample_info = [items[_POS_INDEX['u_uid']], items[_POS_INDEX['u_country']]]
    return sample_info, label_dict, score


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
    data = list()
    for key in datalist:
        data.append([key, datalist[key][0], datalist[key][1]])
    return sorted(data, key=lambda x: -x[0])


def format_data_for_auc(data, key):
    datalist = dict()
    for uid, label_dict, score in data:
        label = label_dict[key]
        if score not in datalist:
            datalist[score] = [0, 0]
        if label >= 1.0:
            datalist[score][0] += 1
        else:
            datalist[score][1] += 1
    data = list()
    for key in datalist:
        data.append([key, datalist[key][0], datalist[key][1]])
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
    totalP = sum([x[1] for x in datalist])
    totalN = sum([x[2] for x in datalist])
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


def per_auc(data):
    sorted_data = sorted(data, key=lambda i: i[0])

    res = dict()

    for key in label_conf_dict.keys():

        user_auc_sum = 0
        users = 0
        user_auc_weighted_sum = 0
        samples = 0

        for uid, group in groupby(sorted_data, lambda i: i[0]):
            group = list(group)
            user_samples = len(group)
            if all(i[1][key] < 1.0 for i in group):
                continue
            if all(i[1][key] >= 1.0 for i in group):
                continue
            user_format_data = format_data_for_auc(group, key)
            user_auc = auc_speedup(user_format_data)
            samples += user_samples
            user_auc_weighted_sum += user_samples * user_auc
            users += 1
            user_auc_sum += user_auc
        format_sort_data = format_data_for_auc(sorted_data, key)
        all_auc = auc_speedup(format_sort_data)
        weight_user_auc = 0.0 if samples == 0 else user_auc_weighted_sum / samples
        per_user_auc = 0.0 if users == 0 else user_auc_sum / users
        res[key] = [all_auc, weight_user_auc, per_user_auc]

    return res


USER_COUNTRY = {'COUNTRY_U': 'US', 'COUNTRY_I': 'IN', 'COUNTRY_O': 'OTHER'}
FIELD_CONF = [USER_COUNTRY]
KEYS = ['ALL', 'COUNTRY_U', 'COUNTRY_I', 'COUNTRY_O']

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    label_conf_dict['SVDWELL'][1] = FLAGS.dwell_label_thresh
    label_conf_dict['SVLIKE'][1] = FLAGS.like_label_thresh
    label_conf_dict['SVSHARE'][1] = FLAGS.share_label_thresh
    label_conf_dict['SVFOLLOW'][1] = FLAGS.follow_label_thresh
    # Clean up the model directory if present
    # shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    full_data = read_input(FLAGS.input_feature_data, FLAGS.input_score_data)
    data_dict = {}
    # reg_data = [i for i in full_data if i[0][2] == '1']
    # vis_data = [i for i in full_data if i[0][2] == '0']
    # ava_data = [k for k in [reg_data, vis_data] if k]
    # if not ava_data:
    ava_data = [full_data]

    aucs = []
    for data in ava_data:
        data_dict['ALL'] = [(i[0][0], i[1], i[2]) for i in data]
        for j, conf in enumerate(FIELD_CONF):
            for k, v in conf.iteritems():
                data_dict[k] = [(i[0][0], i[1], i[2]) for i in data if i[0][j + 1] == v]
        for k in KEYS:
            data_cate = data_dict[k]
            # over_all_auc, p_auc1, p_auc2 = per_auc(data_cate)
            for auc_type, auc_list in per_auc(data_cate).items():
                sample_auc = auc_list[0]
                weighted_user_auc = auc_list[1]
                user_auc = auc_list[2]
                aucs.append('%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f' % (
                    FLAGS.project, FLAGS.time_tag, k, auc_type, sample_auc, weighted_user_auc, user_auc))
    print('\n'.join(aucs))
