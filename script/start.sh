#!/usr/bin/env bash

model="$1"
s_date="$2"
s_hour="$3"
e_date="$4"
e_hour="$5"
user="$6"

if [[ "$#" -ne 6 ]]; then
  echo "usage: bash -x *.sh model s_date s_hour e_date e_hour user"
  exit 1
fi

FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${FILE_DIR}/environment.sh"

export DIR_USER="${PROJECT_DIR}/user/${user}"

export DIR_LOG="${DIR_USER}/${model}/log"
export DIR_METRICS="${DIR_USER}/${model}/metrics"
export DIR_EXPORT="${DIR_USER}/${model}/export"
export DIR_CKPT="${DIR_USER}/${model}/ckpt"
export DIR_PRED="${DIR_USER}/${model}/pred"

mkdir -p ${DIR_LOG}
mkdir -p ${DIR_METRICS}
mkdir -p ${DIR_EXPORT}
mkdir -p ${DIR_CKPT}
mkdir -p ${DIR_PRED}

#training_data_dir="${DIR_SHARE_SAMPLES}/shortVideoBase"
training_data_dir="s3://liveme.data.datastore/shortVideoBase/${address}"
file_log_train_hourly="${DIR_LOG}/train_hourly.log"

declare -A train_hourly_map=(
#["liveme"]="${SCRIPT_DIR}/train_hourly_liveme.sh"
["liveme"]="${SCRIPT_DIR}/train_liveme_daily.sh"
["cheez"]="${SCRIPT_DIR}/train_cheez_daily.sh"
["default"]="${SCRIPT_DIR}/train_cheez_daily.sh"
["test"]="${SCRIPT_DIR}/train_cheez_daily.sh"
["esb_eval"]="${SCRIPT_DIR}/esb_eval_daily.sh"
 )

script_train=${train_hourly_map["${train_script_name}"]}

nohup bash -x ${script_train} ${model}  ${s_date} ${s_hour} ${e_date} ${e_hour} ${training_data_dir}/basicDataALL >>${file_log_train_hourly} &
