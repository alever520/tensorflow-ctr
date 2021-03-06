#!/usr/bin/env bash


#check some commands

declare -a commands_to_check=("shuf" "gzip" "hadoop")

## now loop through the above array
for command in "${commands_to_check[@]}"
do
   type ${command}
   if [[ "$?" -ne 0 ]]; then
      echo "not found command: ${command}"
      exit -1
   fi
   # or do whatever with individual element of the array
done


FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${FILE_DIR}/environment.sh"

PROJ_NAME="download_liveme_data_for_tensorflow"

python_parser="${PYTHON_DIR}/yzy/data/parser/liveme_basic_data.py"

liveme_s3_data_base_path="s3://liveme.data.datastore/PRData/UP/basicData"
liveme_local_output_data_base_path="${PROJECT_DIR}/tmp/training_samples/shortVideoBase"
liveme_local_output_all_data_base_path="${liveme_local_output_data_base_path}/basicDataAll"


if [[ "$#" -ne 4 ]]; then
  echo "usage: bash -x *.sh s_date s_hour e_date e_hour"
  exit 1
fi

s_date="$1"
s_hour="$2"
e_date="$3"
e_hour="$4"


if [[ -z "$e_date" ]]; then
  e_date=20200101
fi
if [[ -z "$e_hour" ]]; then
  e_hour=00
fi


#featureConf="/home/yezhengyuan/shell/features.conf"
#countryConf="/home/yezhengyuan/shell/countries.conf"
#countryPos="16"
#uidPos="4"


UNIQUE=(3 4 5 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 28 29 30 31 32 33)          # Array containing columns to be printed
fields=$( IFS="\t" echo "${UNIQUE[@]}")      # Get the fields in comma-delimited form
# seq -s, 10 would print the string: 1,2,3,4,5,6,7,8,9,10
#seq -s, 10 | cut -d, -f"${fields[@]}"     # Print the desired fields
#printFeatures='$1,$2,$3'

t_time=${s_date}${s_hour}
e_time=${e_date}${e_hour}
while [[ "$t_time" -le "$e_time" ]]
do
  t_date=${t_time:0:8}
  t_hour=${t_time:8:2}

  liveme_s3_all_data_hourly_path="${liveme_s3_data_base_path}/${t_date}/${t_hour}/data"
  liveme_local_output_all_data_hourly_path="${liveme_local_output_all_data_base_path}/${t_date}/${t_hour}"

  check_s3_data_ready ${liveme_s3_all_data_hourly_path}/_SUCCESS 288 "5m" ${PROJ_NAME}
  mkdir -p "${liveme_local_output_all_data_base_path}/${t_date}"

  try_max=5
  try_num=0 
  while [[ try_num -lt ${try_max} ]]
  do
    hadoop fs -text ${liveme_s3_all_data_hourly_path}/part-* 2>/dev/null | python ${python_parser} > ${liveme_local_output_all_data_hourly_path}
    if [[ "$?" -ne 0 ]]; then
      echo "[E] download"
      ((try_num+=1))
      bash -x ${script_monitor} ${TELEPHONE} "[E] $PROJ_NAME $t_time failed ${try_num} times."
    else
      shuf ${liveme_local_output_all_data_hourly_path} | gzip > ${liveme_local_output_all_data_hourly_path}.shuffled.gz
#      cat ${liveme_local_output_register_data_hourly_path} | gzip > ${liveme_local_output_register_data_hourly_path}.gz
      rm ${liveme_local_output_all_data_hourly_path}
      touch ${liveme_local_output_all_data_hourly_path}_SUCCESS
      break
    fi
  done
  if [[ try_num -ge ${try_max} ]]
  then
    bash -x ${script_monitor} ${TELEPHONE} "[E] $PROJ_NAME $t_time failed"
    exit -1
  fi

t_time=$(date -d "${t_date} ${t_hour} +1 hour" +%Y%m%d%H)
done
