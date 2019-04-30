#!/usr/bin/env bash




python_calauc="calAuc.py"
time_tag="2018"
input_feature_data=""
auc_multi_label_json_file=""

export python_auc_args=" \
  --auc_multi_label_json $auc_multi_label_json_file \
  --is_input_gzip false \
  --is_input_tfrecord false \
  --is_input_feature_score_merged true \
  --is_cal_rmse false \
"

python  ${python_calauc} --project exp_range --time_tag $time_tag --input_feature_data ${input_feature_data} ${python_auc_args} | awk '{print $0}'