"""tensorflow for ctr prediction"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import time
import json
from .tf_local.export import export_feature_preprocess
from .yzy.tensorflow.python.estimator.canned.dnn_gpu import DNNClassifierGpu
from .yzy.tensorflow.python.estimator.canned.dnn_yzy import DNNMultiLabelClassifier
from .zbw.model.deepfm import DeepFM

import tensorflow as tf
from tensorflow.contrib.data.python.ops.batching import map_and_batch
from tensorflow.contrib.opt.python.training import lazy_adam_optimizer
from tensorflow.python.training import adagrad, gradient_descent, adam, rmsprop, ftrl, adadelta
import math
import numpy as np
from random import shuffle

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_CSV_COLUMNS = [
]

_CSV_COLUMN_DEFAULTS = [
]

_LABEL_INFO = [
]

input_raw_feature_params = None

feature_layer_params = None

model_training_conf = None


def feature_name_init():
    """read feature json config and parse"""
    global input_raw_feature_params
    global feature_layer_params
    input_raw_feature_params = json.load(open(FLAGS.feature_json))
    feature_layer_params = json.load(open(FLAGS.cross_feature_json))

    for elem in input_raw_feature_params:
        _CSV_COLUMNS.append(elem['name'])

        format_csv_type = elem["format"]["csv"]["type"]
        if format_csv_type == 'float':
            _CSV_COLUMN_DEFAULTS.append([0.0])
        elif format_csv_type == 'string':
            _CSV_COLUMN_DEFAULTS.append([''])
        elif format_csv_type == 'string_list':
            _CSV_COLUMN_DEFAULTS.append([''])
        elif format_csv_type == 'float_list':
            _CSV_COLUMN_DEFAULTS.append([0.0])
        elif format_csv_type == 'int64':
            _CSV_COLUMN_DEFAULTS.append([0])
        elif format_csv_type == 'int64_list':
            _CSV_COLUMN_DEFAULTS.append([0])
        if 'label' in elem:
            elem['weight_cal_fun'] = make_weight_cal_fun(elem)
            elem['multi_label_fun'] = make_multi_label_fun(elem)
            _LABEL_INFO.append(elem)


def model_training_init():
    """read model training json config"""
    global model_training_conf
    model_training_conf = json.load(open(FLAGS.model_training_json))


def make_weight_cal_fun(dict_label):
    kw_label = dict_label["label"]
    kw_weight = kw_label["weight"]
    val_max = kw_label['max']
    val_min = kw_label['min']
    val_denominator = kw_label['denominator']
    label_name = dict_label["name"]
    weight_cond_features = kw_label["weight_cond_features"]
    is_weight_cond_features_exist = weight_cond_features is not None
    split_weight_cond_features = weight_cond_features.split(",") if is_weight_cond_features_exist else None
    label_default_weight = kw_weight[""]
    val_denominator_repo = 1.0 / val_denominator

    def weight_cal_fun(features):
        label_value_in_features = features[label_name]
        after_max_min_denominator_value = tf.maximum(tf.minimum(label_value_in_features, val_max),
                                                     val_min) * val_denominator_repo

        tensor_label_weight = label_default_weight

        if is_weight_cond_features_exist:
            tensor_key = tf.string_join([features[field] for field in split_weight_cond_features], separator="\t")
            for exist_key in kw_weight:
                tensor_label_weight = tf.cond(tf.equal(tensor_key, exist_key), lambda: kw_weight[exist_key],
                                              lambda: tensor_label_weight)
        # if weight_cond_features is not None:
        #     # tensor_key = tf.string_join([features.pop(field) for field in weight_cond_features.split(",")], separator="\t")
        #     for exist_key in kw_weight:
        #         tensor_label_weight = tf.cond(tf.equal(exist_key, exist_key), lambda: kw_weight[exist_key], lambda: label_default_weight)

        return after_max_min_denominator_value * tensor_label_weight

    return weight_cal_fun


def make_multi_label_fun(dict_label):
    kw_label = dict_label["label"]
    kw_weight = kw_label["weight"]
    val_max = kw_label['max']
    val_min = kw_label['min']
    val_denominator = kw_label['denominator']
    label_name = dict_label["name"]
    weight_cond_features = kw_label["weight_cond_features"]
    is_weight_cond_features_exist = weight_cond_features is not None
    split_weight_cond_features = weight_cond_features.split(",") if is_weight_cond_features_exist else None
    label_default_weight = kw_weight[""]
    val_denominator_repo = 1.0 / val_denominator

    def multi_label_fun(features):
        label_value_in_features = features[label_name]
        after_max_min_denominator_value = tf.maximum(tf.minimum(label_value_in_features, val_max),
                                                     val_min) * val_denominator_repo
        return tf.cond(tf.greater_equal(after_max_min_denominator_value, 1.0), lambda: 1, lambda: 0)

    return multi_label_fun


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--is_label_weight", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="using label weight or not")

parser.add_argument(
    '--pos_weight_scale', type=float, default=1.0, help='pos_weight_scale')

parser.add_argument(
    '--neg_weight_scale', type=float, default=1.0, help='neg_weight_scale')

parser.add_argument("--tf_record", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="tf_record sub program")

parser.add_argument("--is_exporting_with_feature_preprocess", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_exporting_with_feature_preprocess")

parser.add_argument(
    '--train_data_dir', type=str, required=True, default='/tmp',
    help='train_data_dir')

parser.add_argument(
    '--train_data_begin_date_hour', type=str, required=True, default='2001010100',
    help='train_data_begin_date_hour')

parser.add_argument(
    '--train_data_end_date_hour', type=str, required=True, default='2001010100',
    help='train_data_end_date_hour')

parser.add_argument(
    '--train_data_dir_postfix', type=str, required=True, default='.shuffled.gz',
    help='train_data_dir_postfix')

parser.add_argument(
    '--test_data_dir', type=str, required=True, default='/tmp',
    help='test_data_dir')

parser.add_argument(
    '--test_data_begin_date_hour', type=str, required=True, default='2001010100',
    help='test_data_begin_date_hour')

parser.add_argument(
    '--test_data_end_date_hour', type=str, required=True, default='2001010100',
    help='test_data_end_date_hour')

parser.add_argument(
    '--test_data_dir_postfix', type=str, required=True, default='.shuffled.gz',
    help='test_data_dir_postfix')

parser.add_argument(
    '--predict_output_dir', type=str, required=True, default='/tmp/pred',
    help='predict_output_dir')

parser.add_argument(
    '--tf_record_input', type=str, default='/tmp',
    help='using tfrecord input format')

parser.add_argument(
    '--tf_record_output', type=str, default='/tmp',
    help='tf_record_output dir for tfrecord subprogram')

parser.add_argument(
    '--model_dir', type=str, default='/tmp',
    help='Base directory for the model.')

parser.add_argument("--is_input_training_files_to_shuffle", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_input_training_files_to_shuffle.")

parser.add_argument("--is_input_training_files_to_reverse_order", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_input_training_files_to_reverse_order.")

parser.add_argument("--is_input_tf_record_format", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_input_tf_record_format.")

parser.add_argument("--is_model_training", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_model_training.")

parser.add_argument("--is_model_training_shuffling", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_model_training_shuffling.")

parser.add_argument("--is_model_evaluating", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_model_evaluating.")

parser.add_argument("--is_model_exporting", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_model_exporting.")

parser.add_argument("--is_model_predicting_test_data", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_model_predicting_test_data.")

parser.add_argument("--is_model_predicting_train_data", type=str2bool, nargs='?',
                    const=True, default='false',
                    help="is_model_predicting_train_data.")

parser.add_argument(
    '--model_export_dir', type=str, default='/tmp/model_export_dir',
    help='Base directory for the model export.')

parser.add_argument(
    '--train_epochs', type=int, default=1, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=256, help='Number of examples per batch.')

parser.add_argument(
    '--dataset_shuffle_buffer_size', type=int, default=10000000, help='dataset_shuffle_buffer_size')

parser.add_argument(
    '--dataset_map_num_parallel_calls', type=int, default=16, help='dataset_map_num_parallel_calls')

parser.add_argument(
    '--predict_batch_size', type=int, default=32768, help='Number of examples per batch for predicting.')

parser.add_argument(
    '--feature_json', type=str, default='/tmp/feature.json',
    help='Path to the feature json.')

parser.add_argument(
    '--model_training_json', type=str, default='/tmp/model_training.json',
    help='Path to the model training json.')

parser.add_argument(
    '--cross_feature_json', type=str, default='/tmp/cross_feature.json',
    help='Path to the cross feature json.')


# parser.add_argument('--hidden_units', nargs='+', type=int, help='<Required> Set flag', required=True)


def build_model_columns(input_raw_feature, feature_layer_feature):
    """Builds a set of wide, fm and deep feature columns."""

    # 生成交叉特征
    def cross_column_parser(config):
        crossed_columns_list = [feature_layer[to_cross_column["column"]][to_cross_column["name"]] for to_cross_column in
                                config["come_from"]]
        hash_bucket_size = config["hash_bucket_size"]
        return tf.feature_column.crossed_column(crossed_columns_list, hash_bucket_size=hash_bucket_size)

    # 用特征列来存储特征，不同数据类型的特征需要用不同的特征列
    def make_feature_layer(feature_layer, column_type):
        if column_type in feature_layer_feature:
            # 数值列
            if column_type == "numeric_column":
                feature_layer[column_type] = {key: tf.feature_column.numeric_column(
                    feature_layer[val["come_from"]["column"]][val["come_from"]["name"]],
                    shape=val.get("shape", [1]),
                    default_value=val.get("default_value", None)) for key, val in
                    feature_layer_feature[column_type].iteritems()}
            # 分桶列
            elif column_type == "bucketized_column":
                feature_layer[column_type] = {key: tf.feature_column.bucketized_column(
                    feature_layer[val["come_from"]["column"]][val["come_from"]["name"]],
                    boundaries=np.arange(val["boundaries_para"]["start"], val["boundaries_para"]["end"],
                                         val["boundaries_para"]["step"]).tolist() if "boundaries_para" in val else val[
                        "boundaries"])
                    for key, val in
                    feature_layer_feature[column_type].iteritems()}
            # 分类词汇列 字符串与证书的映射关系 枚举的方式
            elif column_type == "categorical_column_with_vocabulary_list":
                feature_layer[column_type] = {
                    key: tf.feature_column.categorical_column_with_vocabulary_list(
                        feature_layer[val["come_from"]["column"]][val["come_from"]["name"]],
                        vocabulary_list=val["vocabulary_list"], num_oov_buckets=val.get("num_oov_buckets", 0),
                        default_value=val.get("default_value", -1)) for key, val in
                    feature_layer_feature[column_type].iteritems()}
            # 分类标识列 类似于分桶列 只是以输入的标识作为一个桶
            elif column_type == "categorical_column_with_identity":
                feature_layer[column_type] = {
                    key: tf.feature_column.categorical_column_with_identity(
                        feature_layer[val["come_from"]["column"]][val["come_from"]["name"]],
                        num_buckets=val["num_buckets"],
                        default_value=val.get("default_value", None)) for key, val in
                    feature_layer_feature[column_type].iteritems()}
            # 分类词汇列 词汇映射关系存在文件中 而非一个list列表中
            elif column_type == "categorical_column_with_vocabulary_file":
                vob_dir = feature_layer_feature["categorical_column_with_vocabulary_file_dir"]
                if "categorical_column_with_vocabulary_file" in feature_layer_feature:
                    feature_layer[column_type] = {key: tf.feature_column.categorical_column_with_vocabulary_file(
                        feature_layer[val["come_from"]["column"]][val["come_from"]["name"]],
                        vocabulary_file=str(vob_dir + "/" + val["come_from"]["name"] + ".vob"),
                        vocabulary_size=val.get("vocabulary_size", None), num_oov_buckets=val.get("num_oov_buckets", 0))
                        for key, val in
                        feature_layer_feature[column_type].iteritems()}
            # 分类哈希列 将输入的哈希值 分桶
            elif column_type == "categorical_column_with_hash_bucket":
                feature_layer[column_type] = {
                    key: tf.feature_column.categorical_column_with_hash_bucket(
                        feature_layer[val["come_from"]["column"]][val["come_from"]["name"]],
                        hash_bucket_size=val["hash_bucket_size"]) for key, val in
                    feature_layer_feature[column_type].iteritems()}
            # 分类权重列 用于一个分类id带有数值的情况 ID:value ID用于构成分类列 value则映射成ID的权重
            elif column_type == "weighted_categorical_column":
                feature_layer[column_type] = {key: tf.feature_column.weighted_categorical_column(
                    categorical_column=feature_layer[val["come_from"]["column"]][val["come_from"]["name"]],
                    weight_feature_key=val["weight_feature_key"]) for key, val in feature_layer_feature[
                    column_type].iteritems()}
            # 交叉列
            elif column_type == "crossed_column":
                feature_layer[column_type] = {key: cross_column_parser(val) for key, val in
                                              feature_layer_feature[column_type].iteritems()}
            # 指针列 热独方式处理 维度很高
            elif column_type == "indicator_column":
                feature_layer[column_type] = {key: tf.feature_column.indicator_column(
                    feature_layer[val["come_from"]["column"]][val["come_from"]["name"]]) for key, val in
                    feature_layer_feature[column_type].iteritems()}
            # 嵌入列 自行决定维度和combiner方式
            elif column_type == "embedding_column":
                feature_layer[column_type] = {key: tf.feature_column.embedding_column(
                    feature_layer[val["come_from"]["column"]][val["come_from"]["name"]], dimension=val["dimension"],
                    combiner="mean" if "combiner" not in val else val["combiner"],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1 / math.sqrt(
                        val["dimension"])) if "initializer" not in val else tf.truncated_normal_initializer(
                        mean=val["initializer"]["mean"], stddev=val["initializer"]["stddev"])) for key, val in
                    feature_layer_feature[column_type].iteritems()}
            elif column_type == "fm_embedding_column":
                feature_layer[column_type] = {key: tf.feature_column.embedding_column(
                    feature_layer[val["come_from"]["column"]][val["come_from"]["name"]], dimension=val["dimension"],
                    combiner="mean" if "combiner" not in val else val["combiner"],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1 / math.sqrt(val["dimension"])) if
                    "initializer" not in val else tf.truncated_normal_initializer(mean=val["initializer"]["mean"],
                                                                                  stddev=val["initializer"]["stddev"]))
                    for key, val in feature_layer_feature[column_type].iteritems()}
            elif column_type == "shared_embedding_columns":
                feature_layer[column_type] = {col_name: tf_col for key, val in
                                              feature_layer_feature[column_type].iteritems()
                                              for col_name, tf_col in
                                              zip(key.split(","), tf.feature_column.shared_embedding_columns(
                                                  [feature_layer[cas["column"]][cas["name"]] for cas in
                                                   val["come_from"]],
                                                  dimension=val["dimension"],
                                                  combiner="mean" if "combiner" not in val else val["combiner"],
                                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                              stddev=1 / math.sqrt(
                                                                                                  val["dimension"]))
                                                  if "initializer" not in val else tf.truncated_normal_initializer(
                                                      mean=val["initializer"]["mean"],
                                                      stddev=val["initializer"][
                                                          "stddev"])))}
            else:
                pass
        else:
            pass

    # note the order in the list, some columns has interdependence
    # 特征列生成过程中，有些不能通过原始数据直接生成，需要先转换成某种特征列，在以此类型为输入，获得需要的输出特征列
    # 例如 weighted_categorical-column 需要以 categorical_column_with ... 为输入
    support_column_types = ["numeric_column", "bucketized_column", "categorical_column_with_vocabulary_list",
                            "categorical_column_with_identity", "categorical_column_with_vocabulary_file",
                            "categorical_column_with_hash_bucket", "weighted_categorical_column", "crossed_column",
                            "indicator_column", "embedding_column", "fm_embedding_column", "shared_embedding_columns"]

    feature_layer = {}
    feature_layer["raw_column"] = {raw_feature["name"]: raw_feature["name"] for raw_feature in input_raw_feature}

    # note: should call make_feature_layer in order of support_column_types because of the feature column interdependence
    # 处理特征成特征列 特征列形式的特征存在feature_layer中 以特征列类型为key（第一层key）
    for column_type in support_column_types:
        make_feature_layer(feature_layer, column_type)

    def model_input_parser(model_input_column_value):

        def column_list_func(come_from_column_type, come_from_column_list):
            # 取出与come_from_column_type相同类型的特征列的数据
            this_feature_layer = feature_layer[come_from_column_type]
            # 取出该类型特征数据中 key包含在come_from_column_list这个列表中的特征
            return [this_feature_layer[column_name] for column_name in come_from_column_list]
        # 取出model_input_column_value中包含的所有特征列类型和所有特征名的所有特征， 返回一个extend的list
        return [column for come_from_column_type, come_from_column_list in model_input_column_value.iteritems() for
                column in column_list_func(come_from_column_type, come_from_column_list)]

        # column_list = []
        # for come_from_column_type, come_from_column_list in model_input_column_value.iteritems():
        #     this_feature_layer = feature_layer[come_from_column_type]
        #     column_list.extend([this_feature_layer[column_name] for column_name in come_from_column_list])
        # return column_list
    # 按照feature_layer_feature["model_input"] 中的不同来源取出经过特征列处理的特征
    model_input_column_dict = {model_input_key: model_input_parser(model_input_column_value) for
                               model_input_key, model_input_column_value in
                               feature_layer_feature["model_input"].iteritems()}

    wide_columns = model_input_column_dict["wide_column"]
    fm_first_column = model_input_column_dict["fm_first_column"]
    fm_second_column = model_input_column_dict["fm_second_column"]
    deep_columns = model_input_column_dict["deep_column"]

    return wide_columns, fm_first_column, fm_second_column, deep_columns


_OPTIMIZER_CLS_NAMES = {
    'Adagrad': adagrad.AdagradOptimizer,
    'Adam': adam.AdamOptimizer,
    'Ftrl': ftrl.FtrlOptimizer,
    'RMSProp': rmsprop.RMSPropOptimizer,
    'SGD': gradient_descent.GradientDescentOptimizer,
    'LazyAdam': lazy_adam_optimizer.LazyAdamOptimizer,
}


def build_estimator(model_dir):
    """Build an tensorflow estimator from global json config and save parameters in model_dir.

    Args:
      model_dir: directory where model parameters, graph, etc are saved. If
        `PathLike` object, the path will be resolved. If `None`, will use a
        default value set by the Estimator.
    """

    # get feature columns
    wide_columns, fm_first_column, fm_second_column, deep_columns = build_model_columns(input_raw_feature_params,
                                                                                        feature_layer_params)

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    # kw_run_config = model_training_conf["estimator_run_config"]
    run_config = tf.estimator.RunConfig(
        **model_training_conf["estimator_run_config"]
        # save_checkpoints_steps=kw_run_config["save_checkpoints_steps"],
        #                                 save_checkpoints_secs=kw_run_config["save_checkpoints_secs"],
        #                                 keep_checkpoint_max=kw_run_config["keep_checkpoint_max"],
        #                                 save_summary_steps=kw_run_config["save_summary_steps"],
        #                                 log_step_count_steps=kw_run_config["log_step_count_steps"]
    ).replace(
        session_config=tf.ConfigProto(device_count=model_training_conf["session_config"]["device_count"]))

    model_training_type = model_training_conf["type"]

    DeepfmParams = {
        "embedding_size": fm_second_column[0]._variable_shape.num_elements() if len(fm_second_column) > 0 else None,
        "field_size": len(fm_second_column)
    }

    def make_optimizer(type, conf):
        return _OPTIMIZER_CLS_NAMES[type](**model_training_conf['optimizer'][type][conf])

    if model_training_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            weight_column=model_training_conf[model_training_type]['weight_column'],
            optimizer=make_optimizer(**model_training_conf[model_training_type]['optimizer']),
            config=run_config)
    elif model_training_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            weight_column=model_training_conf[model_training_type]['weight_column'],
            hidden_units=model_training_conf[model_training_type]['hidden_units'],
            dropout=model_training_conf[model_training_type]['dropout'],
            optimizer=make_optimizer(**model_training_conf[model_training_type]['optimizer']),
            config=run_config)
    elif model_training_type == 'dnn_multi_label_classifier':
        return DNNMultiLabelClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            n_classes=model_training_conf[model_training_type]['n_classes'],
            is_multi_label=model_training_conf[model_training_type]['is_multi_label'],
            weight_column=model_training_conf[model_training_type]['weight_column'],
            hidden_units=model_training_conf[model_training_type]['hidden_units'],
            dropout=model_training_conf[model_training_type]['dropout'],
            optimizer=make_optimizer(**model_training_conf[model_training_type]['optimizer']),
            config=run_config)
    elif model_training_type == 'deep_gpu':
        return DNNClassifierGpu(
            model_dir=model_dir,
            feature_columns=deep_columns,
            weight_column=model_training_conf[model_training_type]['weight_column'],
            hidden_units=model_training_conf[model_training_type]['hidden_units'],
            dropout=model_training_conf[model_training_type]['dropout'],
            optimizer=make_optimizer(**model_training_conf[model_training_type]['optimizer']),
            config=run_config)
    elif model_training_type == 'dnn_linear_combined_regressor':
        return tf.estimator.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=make_optimizer(**model_training_conf[model_training_type]['linear_optimizer']),
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=model_training_conf[model_training_type]['dnn_hidden_units'],
            dnn_dropout=model_training_conf[model_training_type]['dnn_dropout'],
            dnn_optimizer=make_optimizer(**model_training_conf[model_training_type]['dnn_optimizer']),
            weight_column=model_training_conf[model_training_type]['weight_column'],
            loss_reduction=model_training_conf[model_training_type]['loss_reduction'],
            config=run_config)
    elif model_training_type == 'deepfm':
        return DeepFM(
            model_dir=model_dir,
            fm_first_feature_columns=fm_first_column,
            fm_second_feature_columns=fm_second_column,
            dnn_feature_columns=deep_columns,
            weight_column=model_training_conf[model_training_type]['weight_column'],
            dnn_hidden_units=model_training_conf[model_training_type]['dnn_hidden_units'],
            embedding_size=DeepfmParams['embedding_size'],
            field_size=DeepfmParams['field_size'],
            linear_optimizer=make_optimizer(**model_training_conf[model_training_type]['linear_optimizer']),
            dnn_optimizer=make_optimizer(**model_training_conf[model_training_type]['dnn_optimizer']),
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=make_optimizer(**model_training_conf[model_training_type]['linear_optimizer']),
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=model_training_conf[model_training_type]['dnn_hidden_units'],
            dnn_dropout=model_training_conf[model_training_type]['dnn_dropout'],
            dnn_optimizer=make_optimizer(**model_training_conf[model_training_type]['dnn_optimizer']),
            weight_column=model_training_conf[model_training_type]['weight_column'],
            config=run_config)


def input_fn(data_files, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    # for data_file in data_files:
    #     assert tf.gfile.Exists(data_file), (
    #         '%s not found. Please make sure you have either run data_download.py or '
    #         'set both arguments --train_data and --test_data.' % data_file)

    model_training_type = model_training_conf["type"]
    weight_column = model_training_conf[model_training_type]['weight_column']
    dataset = None
    is_label_probability = all(
        'as_probability' in dict_label['label'] and dict_label['label']['as_probability'] for dict_label in _LABEL_INFO)

    def features_transform(features):
        sample_weight_origin = sum(
            [dict_label['weight_cal_fun'](features) for dict_label in _LABEL_INFO])
        # sample_weight_origin = sum(map(cal_sample_weight_origin, _LABEL_INFO))
        if weight_column is not None:
            # up weight some negative sample to 1.0,sample_weight_origin is a tensor
            features[weight_column] = tf.cond(sample_weight_origin >= 1.0,
                                              lambda: sample_weight_origin * FLAGS.pos_weight_scale,
                                              lambda: FLAGS.neg_weight_scale)

        if "is_multi_label" in model_training_conf[model_training_type]:
            multi_labels = tf.stack([dict_label['multi_label_fun'](features) for dict_label in _LABEL_INFO])
            return features, multi_labels
        elif is_label_probability:
            if "regressor" in model_training_type:
                return features, tf.stack([features[dict_label['name']] for dict_label in _LABEL_INFO])
            else:
                return features, tf.stack(
                    [tf.greater_equal(features[dict_label['name']], tf.random_uniform(shape=(), minval=0, maxval=1)) for
                     dict_label in _LABEL_INFO])
        else:
            return features, tf.stack([tf.greater_equal(sample_weight_origin, 1.0)])

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, **model_training_conf["odd"]["decode_csv"])
        features = dict(zip(_CSV_COLUMNS, columns))
        return features_transform(features)
        # return features, tf.greater_equal(labels, FLAGS.dwell_label_thresh)

    # Extract lines from input files using the Dataset API.

    if FLAGS.is_input_tf_record_format:
        feature_spec = {}
        type_to_dtype = {
            "fixed_len_bytes": tf.string,
            "fixed_len_int64": tf.int64,
            "fixed_len_float": tf.float32
        }

        type_to_default_value = {
            "fixed_len_bytes": tf.constant("", tf.string),
            "fixed_len_int64": tf.constant(0, tf.int64),
            "fixed_len_float": tf.constant(0.0, tf.float32),
        }

        type_to_dtype_var_len = {
            "var_len_bytes": tf.string,
            "var_len_int64": tf.int64,
            "var_len_float": tf.float32
        }
        for elem in input_raw_feature_params:
            if elem['used'] == 'true' or "label" in elem:
                format_tfrecord_type = elem["format"]["tfrecord"]["type"]
                format_tfrecord_shape = elem["format"]["tfrecord"]["shape"]
                feature_name = elem["name"]
                if format_tfrecord_type in type_to_dtype.keys():
                    feature_spec[feature_name] = tf.FixedLenFeature(format_tfrecord_shape,
                                                                    dtype=type_to_dtype[format_tfrecord_type],
                                                                    default_value=tf.fill(format_tfrecord_shape,
                                                                                          type_to_default_value[
                                                                                              format_tfrecord_type]))
                elif format_tfrecord_type in type_to_dtype_var_len.keys():
                    feature_spec[feature_name] = tf.VarLenFeature(dtype=type_to_dtype_var_len[format_tfrecord_type])

        dataset = tf.data.TFRecordDataset(filenames=data_files, **model_training_conf["odd"]["input_dataset_conf"])

        # Use `tf.parse_single_example()` to extract data from a `tf.Example`
        # protocol buffer, and perform any additional per-record preprocessing.
        def tf_record_dataset_parser(record):
            # keys_to_features = {
            #     "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
            #     "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
            #     "label": tf.FixedLenFeature((), tf.int64,
            #                                 default_value=tf.zeros([], dtype=tf.int64)),
            # }
            parsed = tf.parse_single_example(record, feature_spec)
            return features_transform(parsed)
            # Perform additional preprocessing on the parsed data.
            # image = tf.image.decode_jpeg(parsed["image_data"])
            # image = tf.reshape(image, [299, 299, 1])
            # label = tf.cast(parsed["label"], tf.int32)
            # return {"image_data": image, "date_time": parsed["date_time"]}, label

        if shuffle:
            dataset = dataset.shuffle(buffer_size=FLAGS.dataset_shuffle_buffer_size)
        dataset = dataset.map(tf_record_dataset_parser, num_parallel_calls=FLAGS.dataset_map_num_parallel_calls)
        # dataset = dataset.apply(map_and_batch(
        #     map_func=tf_record_dataset_parser, batch_size=batch_size, num_parallel_batches=8))
    else:
        dataset = tf.data.TextLineDataset(filenames=data_files, **model_training_conf["odd"]["input_dataset_conf"])
        if shuffle:
            dataset = dataset.shuffle(buffer_size=FLAGS.dataset_shuffle_buffer_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=FLAGS.dataset_map_num_parallel_calls)
        # dataset = dataset.apply(map_and_batch(
        #     map_func=parse_csv, batch_size=batch_size, num_parallel_batches=8))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)

    # return dataset

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def input_fn_from_dataset(dataset):
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


import datetime


# hour_start like '2012060100' , return hours include hour_end
def generate_date_hour_pairs(hour_start, hour_end):
    start = datetime.datetime.strptime(hour_start, "%Y%m%d%H")
    end = datetime.datetime.strptime(hour_end, "%Y%m%d%H")
    date_hours_pairs = []
    tmp = start
    while tmp < end:
        date_hours_pairs.append((tmp.strftime("%Y%m%d"), tmp.strftime("%H")))
        tmp += datetime.timedelta(hours=1)
    return date_hours_pairs


def main(unused_argv):
    # Clean up the model directory if present
    # shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    print(time.time(), "build_estimator")

    model = build_estimator(FLAGS.model_dir)

    def predicting_data(model, input_data_path, output_data_path):
        predictions = model.predict(input_fn=lambda: input_fn(
            [input_data_path], 1, False, FLAGS.predict_batch_size))
        model_training_type = model_training_conf["type"]
        is_label_probability = all(
            'as_probability' in dict_label['label'] and dict_label['label']['as_probability'] for dict_label in
            _LABEL_INFO)
        if "regressor" in model_training_type:
            if is_label_probability:
                list_predictions = map(lambda p: str(min(max(p['predictions'][0], 0.0), 1.0)), predictions)
            else:
                list_predictions = map(lambda p: str(p['predictions'][0]), predictions)
        elif "is_multi_label" in model_training_conf[model_training_type]:
            list_predictions = map(lambda p: "\t".join([str(res) for res in (p['logits'])]), predictions)
        else:
            # list_predictions = map(lambda p: str(p['logistic'][0]), predictions)
            list_predictions = [str(p['logistic'][0]) for p in predictions]
        # for prediction in predictions:
        #     list_predictions.append(str(prediction['logistic'][0]))
        # logistic_ndarray = prediction['logistic']
        # # print(type(logistic_ndarray))
        # # print(logistic_ndarray)
        # #size = logistic_tensor.get_shape().as_list()[0]
        # list_predictions += logistic_ndarray.tolist()
        # # print('size:', len(list_predictions))
        # # print('size', size, sep='\t')
        # # for idx in range(0, size, 1):
        # #     list_predictions.append(str(logistic_tensor[idx]))
        open(output_data_path, "wb").write('\n'.join(list_predictions))
        # open(FLAGS.output_predict_of_test_data_path, "wb").write('\n'.join(map(lambda logistic: str(logistic), list_predictions)))

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        # predict first

        if FLAGS.is_model_predicting_test_data and tf.train.latest_checkpoint(FLAGS.model_dir) is not None:

            print(time.time(), "is_model_predicting_test_data")
            flie_pairs_of_input_data_output_score = [(
                FLAGS.test_data_dir + "/" + date + "/" + hour + FLAGS.test_data_dir_postfix,
                FLAGS.predict_output_dir + "/test." + date + hour) for date, hour
                in generate_date_hour_pairs(FLAGS.test_data_begin_date_hour,
                                            FLAGS.test_data_end_date_hour)]
            for (input_data_file, output_score_file) in flie_pairs_of_input_data_output_score:
                predicting_data(model, input_data_file, output_score_file)
            print(time.time(), "finish is_model_predicting_test_data")

        if FLAGS.is_model_evaluating and tf.train.latest_checkpoint(FLAGS.model_dir) is not None:
            # print(time.time(), "is_model_evaluating")
            # for (input_data_path, output_predict_data_path) in zip(FLAGS.test_data.split('#'),
            #                                                             FLAGS.output_predict_of_test_data_path.split('#')):
            #     results = model.evaluate(input_fn=lambda: input_fn(
            #         [input_data_path], 1, False, FLAGS.batch_size))
            #
            # # Display evaluation metrics
            # print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
            # print('-' * 60)
            #
            # for key in sorted(results):
            #     print('%s: %s' % (key, results[key]))
            print(time.time(), "finish is_model_evaluating")

        if FLAGS.is_model_training:
            print(time.time(), "is_model_training")

            train_data_files = [FLAGS.train_data_dir + "/" + date + "/" + hour + FLAGS.train_data_dir_postfix for
                                date, hour in generate_date_hour_pairs(FLAGS.train_data_begin_date_hour,
                                                                       FLAGS.train_data_end_date_hour)]
            if FLAGS.is_input_training_files_to_reverse_order:
                train_data_files = train_data_files[::-1]
            if FLAGS.is_input_training_files_to_shuffle:
                shuffle(train_data_files)
            print(train_data_files)
            model.train(input_fn=lambda: input_fn(
                train_data_files, FLAGS.epochs_per_eval, FLAGS.is_model_training_shuffling, FLAGS.batch_size))
            print(time.time(), "finish is_model_training")

        if FLAGS.is_model_predicting_train_data and tf.train.latest_checkpoint(FLAGS.model_dir) is not None:
            print(time.time(), "is_model_predicting_train_data")
            flie_pairs_of_input_data_output_score = [(
                FLAGS.train_data_dir + "/" + date + "/" + hour + FLAGS.train_data_dir_postfix,
                FLAGS.predict_output_dir + "/train." + date + hour) for date, hour
                in generate_date_hour_pairs(FLAGS.train_data_begin_date_hour,
                                            FLAGS.train_data_end_date_hour)]
            for (input_data_file, output_score_file) in flie_pairs_of_input_data_output_score:
                predicting_data(model, input_data_file, output_score_file)
            print(time.time(), "finish is_model_predicting_train_data")

        if FLAGS.is_model_exporting:

            print(time.time(), "is_model_exporting")

            feature_spec = {}
            type_to_dtype = {
                "fixed_len_bytes": {
                    "dtype": tf.string,
                    "default_value": [""]
                },
                "fixed_len_int64": {
                    "dtype": tf.int64,
                    "default_value": [0]
                },
                "fixed_len_float": {
                    "dtype": tf.float32,
                    "default_value": [0.0]
                }
            }
            # type_to_default_value = {
            #     "fixed_len_bytes": tf.constant("", tf.string),
            #     "fixed_len_int64": tf.constant(0, tf.int64),
            #     "fixed_len_float": tf.constant(0.0, tf.float32),
            # }
            # type_to_default_value = {
            #     "fixed_len_bytes": [""],
            #     "fixed_len_int64": [0],
            #     "fixed_len_float": [0.0],
            # }
            type_to_dtype_var_len = {
                "var_len_bytes": tf.string,
                "var_len_int64": tf.int64,
                "var_len_float": tf.float32
            }

            for elem in input_raw_feature_params:
                if elem['used'] == 'true':
                    format_tfrecord_type = elem["format"]["tfrecord"]["type"]
                    format_tfrecord_shape = elem["format"]["tfrecord"]["shape"]
                    default_value_num = reduce(lambda x, y: x * y, format_tfrecord_shape) if len(
                        format_tfrecord_shape) != 0 else 1
                    feature_name = elem["name"]
                    if format_tfrecord_type in type_to_dtype.keys():
                        # default_value = type_to_dtype[format_tfrecord_type]["default_value"] * default_value_num
                        default_value = None  # set None to reject prediction when input example missing some features
                        feature_spec[feature_name] = tf.FixedLenFeature(format_tfrecord_shape,
                                                                        dtype=type_to_dtype[format_tfrecord_type][
                                                                            "dtype"],
                                                                        default_value=default_value)
                    elif format_tfrecord_type in type_to_dtype_var_len.keys():
                        feature_spec[feature_name] = tf.VarLenFeature(dtype=type_to_dtype_var_len[format_tfrecord_type])

            if FLAGS.is_exporting_with_feature_preprocess:
                serving_input_receiver_fn = export_feature_preprocess.build_parsing_serving_input_receiver_fn(
                    feature_spec, input_raw_feature_params=input_raw_feature_params)
            else:
                serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

            model.export_savedmodel(FLAGS.model_export_dir, serving_input_receiver_fn)
            print(time.time(), "finish is_model_exporting")


def prepare_tf_record(unused_argv):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(FLAGS.tf_record_output, options=options)

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    get_func_by_raw_feature_type = {
        "num": _float_feature,
        "string": _bytes_feature
    }
    from random import shuffle
    lines = [line.rstrip('\n') for line in open(FLAGS.tf_record_input)]
    shuffle(lines)

    # from tensorflow.core.example import example_pb2
    # example = example_pb2.Example()
    example = tf.train.Example()
    get_repeated_list = {}

    for field in input_raw_feature_params:
        if field["type"] == "num":
            repeated_list = example.features.feature[field["name"]].float_list.value
            repeated_list.append(0.0)
            get_repeated_list[field["name"]] = repeated_list

        elif field["type"] == "int":
            repeated_list = example.features.feature[field["name"]].int64_list.value
            repeated_list.append(0)
            get_repeated_list[field["name"]] = repeated_list
        else:
            repeated_list = example.features.feature[field["name"]].bytes_list.value
            repeated_list.append("")
            get_repeated_list[field["name"]] = repeated_list

    # for field in input_raw_feature_params:
    #     example.features.feature[field["name"]].float_list.value.append(1.0)

    def val_convert_by_format(format, val):
        if format == "num":
            return float(val)
        elif format == "string":
            return val
        else:
            return int(val)

    line_id = 0
    for line in lines:
        line_id += 1
        if line_id % 10000 == 0:
            print("current convert ", line_id)
        arr = line.split("\t")
        count = 0

        for field in input_raw_feature_params:
            get_repeated_list[field["name"]][0] = val_convert_by_format(field["type"], arr[count])
            # example.features.feature[field["name"]] = get_func_by_raw_feature_type[field["type"]](arr[count])
            count += 1
            # feature_holdplace = feature
            # Create an example protocol buffer
            #     example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    options = vars(FLAGS)
    print(options)

    feature_name_init()
    model_training_init()

    if FLAGS.tf_record:
        tf.app.run(main=prepare_tf_record, argv=[sys.argv[0]] + unparsed)
    else:
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
