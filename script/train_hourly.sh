#!/usr/bin/env bash
model_name="$1"
s_date="$2"
s_hour="$3"
e_date="$4"
e_hour="$5"
input_register_data_dir="$6"
input_visitor_data_dir="$7"
input_all_data_dir="$8"


if [[ "$#" -ne 8 ]]; then
  echo "usage: *.sh model_name s_date s_hour e_date e_hour"
  exit 1
fi

if [[ -z "$model_name" ]]; then
  echo "usage: *.sh model_name s_date s_hour e_date e_hour"
  exit 1
fi

if [[ -z "$e_date" ]]; then
  e_date=20200101
fi
if [[ -z "$e_hour" ]]; then
  e_hour=00
fi

PROJ_NAME="tensorflow_model_training_${model_name}"
python_model_training="${PYTHON_DIR}/wide_deep.py"
python_calauc="${PYTHON_DIR}/${python_auc_file}"
file_auc="${DIR_METRICS}/${metrics_auc_file}"

#hadoop fs -mkdir -p ${s3_model_path}

t_time=${s_date}${s_hour}
e_time=${e_date}${e_hour}


declare -A kwargs=(
["model_export_dir"]="$DIR_EXPORT"
 )

inputKwargs=""
for key in "${!kwargs[@]}"; do  inputKwargs=" --$key ${kwargs[$key]} $inputKwargs "; done

# outpath=14days_test1
declare -A kwargsDynamic=(
)

while [[ "$t_time" -le "$e_time" ]]
do
#  t_date_train_data=${t_time:0:8}
#  t_hour_train_data=${t_time:8:2}
  t_date_base=${t_time:0:8}
  t_hour_base=${t_time:8:2}

  t_plus_1_hour=$(date -d "${t_date_base} ${t_hour_base} +1 hour" +%Y%m%d%H)

  t_train_data=$(date -d "${t_date_base} ${t_hour_base} ${train_hour_delta} hour" +%Y%m%d%H)
  t_date_train_data=${t_train_data:0:8}
  t_hour_train_data=${t_train_data:8:2}
  t_test_data=$(date -d "${t_date_base} ${t_hour_base} ${test_hour_delta} hour" +%Y%m%d%H)
  t_date_test_data=${t_test_data:0:8}
  t_hour_test_data=${t_test_data:8:2}


#output_score_data="${prediction_dir}/predict.sv_1hours_${t_date}${t_hour}"
output_log_data="${DIR_LOG}/log.python.train.${t_date_train_data}${t_hour_train_data}"

training_data_file_postfix=".shuffled.gz"
testiing_data_file_postfix=".shuffled.gz"

training_data_file_ready_flag_postfix="_SUCCESS"

declare -a local_data_files_to_check=(
#"${input_register_data_dir}/${t_date_train_data}/${t_hour_train_data}/${training_data_file_ready_flag_postfix}"
#"${input_visitor_data_dir}/${t_date_train_data}/${t_hour_train_data}/${training_data_file_ready_flag_postfix}"
#"${input_register_data_dir}/${t_date_test_data}/${t_hour_test_data}/${training_data_file_ready_flag_postfix}"
#"${input_visitor_data_dir}/${t_date_test_data}/${t_hour_test_data}/${training_data_file_ready_flag_postfix}"
"${input_all_data_dir}/${t_date_train_data}/${t_hour_train_data}/${training_data_file_ready_flag_postfix}"
"${input_all_data_dir}/${t_date_test_data}/${t_hour_test_data}/${training_data_file_ready_flag_postfix}"
)

for file in "${local_data_files_to_check[@]}"
do
   check_s3_data_ready ${file} 72 "5m" ${PROJ_NAME}
done


#train_data_01=${input_register_data_dir}/${t_date_train_data}/${t_hour_train_data}/${training_data_file_postfix}
#train_data_02=${input_visitor_data_dir}/${t_date_train_data}/${t_hour_train_data}/${training_data_file_postfix}
#train_data_for_predict="${train_data_01}#${train_data_02}"
train_data=${input_all_data_dir}/${t_date_train_data}/${t_hour_train_data}/${training_data_file_postfix}
test_data=${input_all_data_dir}/${t_date_test_data}/${t_hour_test_data}/${testiing_data_file_postfix}

#test_data_01=${input_register_data_dir}/${t_date_test_data}/${t_hour_test_data}/${training_data_file_postfix}
#test_data_02=${input_visitor_data_dir}/${t_date_test_data}/${t_hour_test_data}/${training_data_file_postfix}
#test_data="${test_data_01}#${test_data_02}"
#output_predict_of_test_data_path_01="${DIR_PRED}/predict.test.data.register.${t_date_test_data}${t_hour_test_data}"
#output_predict_of_test_data_path_02="${DIR_PRED}/predict.test.data.visitor.${t_date_test_data}${t_hour_test_data}"
output_predict_of_test_data_path="${DIR_PRED}/predict.test.data.all.${t_date_test_data}${t_hour_test_data}"
#output_predict_of_test_data_path="${output_predict_of_test_data_path_01}"
#output_predict_of_train_data_path_01="${DIR_PRED}/predict.train.data.register.${t_date_train_data}${t_hour_train_data}"
#output_predict_of_train_data_path_02="${DIR_PRED}/predict.train.data.visitor.${t_date_train_data}${t_hour_train_data}"
output_predict_of_train_data_path="${DIR_PRED}/predict.train.data.all.${t_date_train_data}${t_hour_train_data}"

#output_predict_of_train_data_path="${output_predict_of_train_data_path_01}"

#check_local_data_ready ${train_data_01}_SUCCESS 72 "5m" ${PROJ_NAME}
#check_local_data_ready ${train_data_02}_SUCCESS 72 "5m" ${PROJ_NAME}
#check_local_data_ready ${test_data_01}_SUCCESS 72 "5m" ${PROJ_NAME}
#check_local_data_ready ${test_data_02}_SUCCESS 72 "5m" ${PROJ_NAME}

kwargsDynamic["train_data"]=${train_data}
kwargsDynamic["train_data_for_predict"]=${train_data}
kwargsDynamic["test_data"]=${test_data}
kwargsDynamic["output_predict_of_test_data_path"]=${output_predict_of_test_data_path}
kwargsDynamic["output_predict_of_train_data_path"]=${output_predict_of_train_data_path}

inputKwargsDynamic=""
for key in "${!kwargsDynamic[@]}"; do  inputKwargsDynamic=" --$key ${kwargsDynamic[$key]} $inputKwargsDynamic "; done

#train_data="${input_data_dir}/${t_date}/${t_hour}"
#test_data="${input_data_dir}/${t_date_predict}/${t_hour_predict}"

if [ "${is_python_training}" == "Y" ];then
  python ${python_model_training} --model_dir ${DIR_CKPT}  $python_train_static_args $inputKwargs $inputKwargsDynamic   >${output_log_data}
  if [[ "$?" -ne 0 ]]; then
    echo "[ERROR] python ${python_model_training} return non zero"
    bash -x ${script_monitor} ${TELEPHONE} "[E] $PROJ_NAME $t_time failed."
    exit -1
  fi
fi

if [ "${is_upload_model_to_s3}" == "Y" ];then

  try_max=5
  try_num=0
  while [[ try_num -lt ${try_max} ]]
  do
    new_ts=$(echo ${DIR_EXPORT}/* | awk ' {print $(NF)}'| sed 's/.*\/\(.*\)/\1/g' )
    if [[ -z "${new_ts}" || "${new_ts}" == "*"  ]]; then
      echo "no new_ts found "
      exit 1
    else
      aws s3 sync ${DIR_EXPORT}/${new_ts} ${s3_model_path}/${new_ts}
    fi

    if [[ "$?" -ne 0 ]]; then
      echo "[ERROR] aws s3 sync"
      ((try_num+=1))
      bash -x ${script_monitor} ${TELEPHONE} "[E] $PROJ_NAME $t_time failed ${try_num} times."
    else
      s3_newest_model_path=$(aws s3 ls ${s3_model_path} | tail -1 | awk '{print $6}')
#      hadoop fs -touchz ${s3_newest_model_path}_SUCCESS
      touch ${s3_newest_model_path}_SUCCESS
      aws s3 cp ${s3_newest_model_path}_SUCCESS s3_model_path
      rm -f ${s3_newest_model_path}_SUCCESS
      break
    fi
  done
  if [[ try_num -ge ${try_max} ]]
  then
    bash -x ${script_monitor} ${TELEPHONE} "[E] $PROJ_NAME $t_time failed"
    exit 1
  fi


fi

if [ -d ${DIR_EXPORT} ];then
  find ${DIR_EXPORT} -mindepth 1 -maxdepth 1 -mtime +0 -type d | xargs rm -rf
fi

if [ -d ${DIR_PRED} ];then
  find ${DIR_PRED} -mindepth 1 -maxdepth 1 -mtime +10 | xargs rm -rf
fi

if [ -d ${DIR_LOG} ];then
  find ${DIR_LOG} -mindepth 1 -maxdepth 1 -mtime +10 | xargs rm -rf
fi
#if [ -f ${kwargsDynamic["test_data"]} ] && [ -f ${kwargsDynamic["output_prediction_path"]} ]; then
#  auc="$(python  calAucV5.py --project $model_name --time_tag $t_predict --input_feature_data ${kwargsDynamic["test_data"]} --input_score_data ${kwargsDynamic["output_prediction_path"]} --dwell_label_thresh $auc_dwell_thresh | awk '{print $0}')"
#  echo -e "${auc}" >> ./metrics/test_auc_${model_name}
#fi

if [ "${is_python_auc}" == "Y" ];then
  if [ "${is_python_train_auc}" == "Y" ];then
    if [ -f ${train_data} ] && [ -f ${output_predict_of_train_data_path} ]; then
      auc="$(python  ${python_calauc} --project ${model_name}_for_register_train --time_tag $t_train_data --input_feature_data ${train_data} --input_score_data ${output_predict_of_train_data_path} ${python_auc_args} | awk '{print $0}')"
      echo -e "${auc}" >> ${file_auc}
    fi
#    if [ -f ${train_data_02} ] && [ -f ${output_predict_of_train_data_path_02} ]; then
#      auc="$(python  ${python_calauc} --project ${model_name}_for_visitor_train --time_tag $t_train_data --input_feature_data ${train_data_02} --input_score_data ${output_predict_of_train_data_path_02} ${python_auc_args} | awk '{print $0}')"
#      echo -e "${auc}" >> ${file_auc}
#    fi
  fi
  if [ "${is_python_test_auc}" == "Y" ];then
    if [ -f ${test_data} ] && [ -f ${output_predict_of_test_data_path} ]; then
      auc="$(python  ${python_calauc} --project ${model_name}_for_register_test --time_tag $t_test_data --input_feature_data ${test_data} --input_score_data ${output_predict_of_test_data_path} ${python_auc_args} | awk '{print $0}')"
      echo -e "${auc}" >> ${file_auc}
    fi
#    if [ -f ${test_data_02} ] && [ -f ${output_predict_of_test_data_path_02} ]; then
#      auc="$(python  ${python_calauc} --project ${model_name}_for_visitor_test --time_tag $t_test_data --input_feature_data ${test_data_02} --input_score_data ${output_predict_of_test_data_path_02} ${python_auc_args} | awk '{print $0}')"
#      echo -e "${auc}" >> ${file_auc}
#    fi
  fi
fi

t_time=${t_plus_1_hour}
done
