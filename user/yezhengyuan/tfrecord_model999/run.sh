#!/usr/bin/env bash
FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
dir_user_workspace="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

user=$(basename ${dir_user_workspace})
model_id=$(basename ${FILE_DIR})

if [[ -z "$user" ]]; then
  echo "[E] user is empty"
  exit 1
fi

export TELEPHONE="18810476634"

entry_script="${FILE_DIR}/../../../script/start.sh"

model="${user}_${model_id}"
s_date="20180501"
s_hour="00"
e_date="20180501"
e_hour="02"

export address="liveme_tfrecord_A_US_gz"
export product="liveme"
export train_script_name="liveme"
feature_json_file="${FILE_DIR}/product_related/${product}/feature/input_raw_feature.json"
model_training_json_file="${FILE_DIR}/model_training.json"
cross_feature_json_file="${FILE_DIR}/product_related/${product}/feature/feature_layer.json"
auc_multi_label_json_file="${FILE_DIR}/product_related/${product}/auc/auc_multi_label.json"

export python_train_static_args=" \
  --is_model_training true \
  --is_model_training_shuffling false \
  --is_model_evaluating false \
  --is_model_exporting true \
  --is_exporting_with_feature_preprocess true \
  --is_model_predicting_train_data false \
  --is_model_predicting_test_data false \
  --is_input_tf_record_format true \
  --train_epochs 1 \
  --epochs_per_eval 1 \
  --batch_size 256 \
  --predict_batch_size 32768 \
  --is_label_weight true \
  --feature_json $feature_json_file \
  --model_training_json $model_training_json_file \
  --cross_feature_json $cross_feature_json_file \
  --pos_weight_scale 1.0 \
  --neg_weight_scale 1.0 \
"

export python_auc_args=" \
  --auc_multi_label_json $auc_multi_label_json_file \
  --is_input_gzip \
  --is_input_tfrecord true \
  --is_cal_rmse false \
  --cross_feature_json $cross_feature_json_file \
"

#  --model_type deep \
#  --hidden_units 128 64 32 16 \
#the training python code will predict first, then train.
#so you must init the model first by enable train but not predict for the first hour
#then you can enable both of training and predicting.
export test_hour_delta="+0"
#current not work
export train_hour_delta="+0"

export t_train_period="+1"
export t_test_period="+1"


export is_python_auc="Y"
export is_python_train_auc="Y"
export is_python_test_auc="Y"
export python_auc_file="calAucV2.py"
export metrics_auc_file="auc.txt"
export is_python_training="Y"
# Y for upload N for not upload
export is_upload_model_to_s3="Y"
export s3_model_path="s3://liveme.data.datastore/PRData/report_forms/tensorflow_serving/model/${model}"

bash -x ${entry_script} ${model} ${s_date} ${s_hour} ${e_date} ${e_hour} ${user}















