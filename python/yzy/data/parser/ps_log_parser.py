import matplotlib
matplotlib.use('Agg')
import seaborn as sns, numpy as np
sns.set();
#np.random.seed(0)
#x = np.random.randn(100)
#sns_plot = sns.distplot(x)
#figure = sns_plot.get_figure()
#figure.savefig('/tmp/yzy.output.png', dpi=400)
import fileinput
import time

# _pos_dict = {
#     "U_PREF": 30,
#     "V_AUDIENCE_NUM": 102
# }
#
#
# length_require = 147
#
# _pos = []
#
# _label_info = list(range(1, 7, 1))
# _user_info = [8, 16, 18, 19, 21]
# _anchor_info = [36, 41]
# _video_info = list(range(127, 147, 1)) + [_pos_dict["U_PREF"], _pos_dict["V_AUDIENCE_NUM"]]
#
# _pos.extend(_label_info)
# _pos.extend(_user_info)
# _pos.extend(_anchor_info)
# _pos.extend(_video_info)
#
# _filter = {
#     "POSID": {
#         "index": 7,
#         "in": {
#             "8002",
#             "9002"
#         }
#     },
#     "ZONE": {
#         "index": 125,
#         "in": {
#             "A_US"
#         }
#     }
# }


# _exp = ["cheez_46", "cheez_45", "cheez_44", "cheez_43"]
# _country = ["US", "IN"]
#
# _pos = {
#     "SVE_EXP": 7,
#     "SVU_COUNTRY": 15,
#     "SVV_COUNTRY": 17
# }
# count_same = dict()
# count_diff = dict()

# for exp in _exp:
#     for country in _country:
#         key = country + "#" + exp
#         count_same[key] = 0
#         count_diff[key] = 0
import json
import pandas as pd

l = []
d = {"predictor_id":[], "cal_d":[]}
for line in fileinput.input():
    line = line.rstrip('\n')
    j = json.loads(line)

    d["predictor_id"].append(j["predictor_id"])
    d["cal_d"].append(j["cal_d"])

df = pd.DataFrame(d)
#print(df)

df["cal_d"] = df["cal_d"].astype(int)//10
#df["times"] = 1
#print(df)


#print(df[['predictor_id', 'cal_d', "times"]].groupby(['predictor_id', 'cal_d']).agg(['count']))

#for predictor_id in ["liveme_tensorflow_dnn_01", "liveme_tensorflow_dnn_02", "sv_tensorflow_dnn_01"]:
for predictor_id in ["liveme_tensorflow_dnn_01"]:
    df_tmp = df[df.predictor_id == predictor_id][["cal_d"]]
    arr = np.histogram(df_tmp, bins=[0, 1, 2, 3, 4, 5, 10, 100, 200, 500, 1200, 1400, 1000000000])
    print("\t".join([str(k) for k in arr[0]]))
    print("\t".join([str(k) for k in arr[1]]))
    sns_plot = sns.distplot(df_tmp[["cal_d"]], kde=False)
    figure = sns_plot.get_figure()
    figure.savefig('/tmp/yzy.ps.log.cal_d.dist.' + predictor_id + '.png', dpi=400)







    # if len(line) < length_require:
    #     continue
    #
    # is_drop = False
    #
    # for k, v in _filter.iteritems():
    #     if line[v["index"]] not in v["in"]:
    #         is_drop = True
    #
    # if is_drop:
    #     continue
    #
    # res = "\t".join([line[i] if line[i] != "None" else "" for i in _pos])
    #
    # # st_str = line[5]
    # # if not st_str.isdigit():
    # #     line_time_tm_hour = ""
    # #     line_time_tm_wday = ""
    # # else:
    # #     line_time = time.gmtime(int(st_str))
    # #     line_time_tm_hour = str(line_time.tm_hour)
    # #     line_time_tm_wday = str((line_time.tm_wday + 1) % 7)
    # #
    # # res += "\t" + line_time_tm_hour + "\t" + line_time_tm_wday
    #
    # print(res)
    #



#     for exp in _exp:
#         for country in _country:
#             key = country + "#" + exp
#             if exp in line[_pos["SVE_EXP"]] and country == line[_pos["SVU_COUNTRY"]]:
#                 if line[_pos["SVU_COUNTRY"]] == line[_pos["SVV_COUNTRY"]]:
#                     count_same[key] += 1
#                 else:
#                     count_diff[key] += 1
#
# print "\t".join(["filter_type", "same_country", "diff_country"])
#
# for key, val in count_same.iteritems():
#     print "\t".join([str(key), str(val), str(count_diff[key])])

