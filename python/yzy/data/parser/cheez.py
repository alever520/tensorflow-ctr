import fileinput
import time

_pos = [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33]



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

    res = "\t".join([line[i-1] if line[i-1] != "None" else "" for i in _pos])

    st_str = line[5]
    if not st_str.isdigit():
        line_time_tm_hour = ""
        line_time_tm_wday = ""
    else:
        line_time = time.gmtime(int(st_str))
        line_time_tm_hour = str(line_time.tm_hour)
        line_time_tm_wday = str((line_time.tm_wday + 1) % 7)

    res += "\t" + line_time_tm_hour + "\t" + line_time_tm_wday

    print(res)




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

