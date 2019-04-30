#! /usr/bin/python

from itertools import groupby
import sys


def readInput():
    for line in sys.stdin:
        items = line.rstrip().split('\t')
        label = int(items[1])
        score = round(float(items[2]), 6)
        sample_info = items[3].split('\6')
        yield sample_info, label, score


def convert_label(value):
    res = 0
    if value > 1:
        res = value
    return res


def rankPercentFun(data):
    length = len(data)
    datalist1 = map(lambda y: ((y[0] + 0.5) / length, convert_label(y[1][1])),
                    list(enumerate(sorted(data, key=lambda x: x[2]))))
    sumDwell = sum(map(lambda x: x[1], datalist1))
    rankPercent = 0.5
    if sumDwell != 0:
        rankPercent = sum(map(lambda x: x[0] * x[1], datalist1)) / sumDwell
    return rankPercent


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


def auc(datalist):
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


def per_auc(data):
    sorted_data = sorted(data, key=lambda i: i[0])
    auc_sum1 = 0
    group_sum1 = 0
    auc_sum = 0
    group_sum = 0

    for uid, group in groupby(sorted_data, lambda i: i[0]):
        group = list(group)
        num = len(group)
        if all(i[1] < int(sys.argv[1]) for i in group):
            continue
        sort_group_data = formatdata(group, int(sys.argv[1]))
        per_auc = auc(sort_group_data)
        group_sum += num
        auc_sum += num * per_auc
        group_sum1 += 1
        auc_sum1 += per_auc
    format_sort_data = formatdata(sorted_data, int(sys.argv[1]))
    all_auc = auc(format_sort_data)

    return all_auc, auc_sum / group_sum, auc_sum1 / group_sum1


def per_rankPercentFun(data):
    sorted_data = sorted(data, key=lambda i: i[0])
    rankPercent_sum1 = 0
    group_sum1 = 0
    rankPercent_sum = 0
    group_sum = 0

    for uid, group in groupby(sorted_data, lambda i: i[0]):
        group = list(group)
        num = len(group)
        per_rankPercent = rankPercentFun(group)
        if per_rankPercent != 0.5:
            group_sum += num
            rankPercent_sum += num * per_rankPercent
            group_sum1 += 1
            rankPercent_sum1 += per_rankPercent
    all_rankPercent = rankPercentFun(sorted_data)

    return all_rankPercent, rankPercent_sum / group_sum, rankPercent_sum1 / group_sum1


GENDER_CONF = {'GENDER_M': '1', 'GENDER_F': '0', 'GENDER_U': '-1'}
ANCHOR_GENDER_CONF = {'A_GENDER_M': '1', 'A_GENDER_F': '0', 'A_GENDER_U': '-1'}
USER_TYPE_CONF = {'NEW_USER': '1', 'OLD_USER': '0'}
FIELD_CONF = [GENDER_CONF, ANCHOR_GENDER_CONF]
KEYS = ['ALL', 'GENDER_M', 'GENDER_F', 'GENDER_U', 'A_GENDER_M', 'A_GENDER_F', 'A_GENDER_U']
UT = 3

if __name__ == "__main__":
    data_iter = readInput()
    all_data = list(data_iter)
    new_user_data = [i for i in all_data if i[0][UT] == '1']
    old_user_data = [i for i in all_data if i[0][UT] == '0']
    aucs = []

    for data in [old_user_data]:
        data_dict = {}
        data_dict['ALL'] = [(i[0][0], i[1], i[2]) for i in data]
        for j, conf in enumerate(FIELD_CONF):
            for k, v in conf.iteritems():
                data_dict[k] = [(i[0][0], i[1], i[2]) for i in data if i[0][j + 1] == v]
        for k in KEYS:
            data_cate = data_dict[k]
            over_all_auc, p_auc1, p_auc2 = per_rankPercentFun(data_cate)
            aucs.append('%.3f %.3f %.3f' % (over_all_auc, p_auc1, p_auc2))
    print
    ' '.join(aucs)
