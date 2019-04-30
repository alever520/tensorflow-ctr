#!/usr/bin/env bash

usage() {
  echo "usage: $0 telephones msg"
  exit 1
}

if [ $# -lt 2 ]; then
  usage
fi


telephone="$1"
msg="$2"
errtype=567


curl -g "http://140.143.118.165:8080/?number=${telephone}&msg=${msg}&errtype=${errtype}" --retry 10 --connect-timeout 5

exit 0;
