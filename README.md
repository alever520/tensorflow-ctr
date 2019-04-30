
#download training samples
nohup bash -x script/downloadCheezDataHourly.sh 20180306 00 20200101 00 &

#how to train model
mkdir -p ${project}/user/${user}

cp -r ${project}/user/yezhengyuan/* ${project}/user/${user}

vim ${project}/user/${user}/run.sh

bash -x ${project}/user/${user}/run.sh >> /your/log/path


###Note:
1. the model training will predict before train, so disable predicting in train_hourly.sh to init the model at first hour



