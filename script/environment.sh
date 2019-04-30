#!/usr/bin/env bash
FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PROJECT_DIR="${FILE_DIR}/.."

export PYTHON_DIR="${PROJECT_DIR}/python"
export SCRIPT_DIR="${PROJECT_DIR}/script"
export DIR_SHARE_SAMPLES="${PROJECT_DIR}/tmp/training_samples"

#export TELEPHONE="13752356970"
export script_monitor="${SCRIPT_DIR}/send_message.sh"

check_s3_data_ready() {
  local data_path="$1"
  local max_try_time="$2"
  local interval="$3"
  local project="$4"
  local try_time=1
  if [[ -z "$max_try_time" ]]; then
    max_try_time=120
  fi
  if [[ -z "$interval" ]]; then
    interval="5m"
  fi
  aws s3 ls ${data_path}
  while [[ "$?" -ne 0 ]]
  do
    if [[ "$try_time" -gt "$max_try_time" ]]; then
      echo "[E] data $data_path not ready in ${max_try_time} multiply ${interval}."
      bash -x ${script_monitor} ${TELEPHONE} "[E] ${project} data $data_path not ready in $((max_try_time/12)) hours."
      exit 1
    fi
    sleep ${interval}
    let ++try_time
    aws s3 ls ${data_path}
  done
  return 0
}

export -f check_s3_data_ready

check_local_data_ready() {
  local data_path="$1"
  local max_try_time="$2"
  local interval="$3"
  local project="$4"
  local try_time=1
  if [[ -z "$max_try_time" ]]; then
    max_try_time=120
  fi
  if [[ -z "$interval" ]]; then
    interval="5m"
  fi
  while [ ! -f "$data_path" ]
  do
    if [[ "$try_time" -gt "$max_try_time" ]]; then
      echo "[E] data $data_path not ready in ${max_try_time} multiply ${interval}."
      bash -x ${script_monitor} ${TELEPHONE} "[E] ${project} data $data_path not ready in $((max_try_time/12)) hours."
      exit 1
    fi
    sleep ${interval}
    let ++try_time
  done
  return 0
}

export -f check_local_data_ready
