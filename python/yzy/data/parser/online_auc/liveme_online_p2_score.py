import fileinput
import time

_pos = []

_label_info = list(range(1, 7, 1))
_user_info = [16, 13]

length_require = 147
exps = ["live_91", "live_92"]

_pos.extend(_label_info)
_pos.extend(_user_info)

_filter = {
    "POSID": {
        "index": 7,
        "in": {
            "8002",
            "9002"
        }
    },
    "ZONE": {
        "index": 125,
        "in": {
            "A_US"
        }
    }
}


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

for line in fileinput.input():
    line = line.rstrip('\n').split('\t')
    if len(line) < length_require:
        continue

    is_drop = False

    for k, v in _filter.iteritems():
        if line[v["index"]] not in v["in"]:
            is_drop = True

    if is_drop:
        continue

    for exp in exps:
        if exp in line[11]:
            res = "\t".join([line[i] if line[i] != "None" else "" for i in _pos]) + "\t" + exp
            print(res)


    # st_str = line[5]
    # if not st_str.isdigit():
    #     line_time_tm_hour = ""
    #     line_time_tm_wday = ""
    # else:
    #     line_time = time.gmtime(int(st_str))
    #     line_time_tm_hour = str(line_time.tm_hour)
    #     line_time_tm_wday = str((line_time.tm_wday + 1) % 7)
    #
    # res += "\t" + line_time_tm_hour + "\t" + line_time_tm_wday






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

