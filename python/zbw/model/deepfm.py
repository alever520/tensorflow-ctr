"""DeepFM training model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import six
import tensorflow as tf
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util

# The default learning rates of dnn and fm model.
_DNN_LEARNING_RATE = 0.001
_FM_LEARNING_RATE = 0.005


def _check_no_sync_replicas_optimizer(optimizer):
  if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
    raise ValueError(
        'SyncReplicasOptimizer does not support multi optimizers case. '
        'Therefore, it is not supported in DeepFM model. '
        'If you want to use this optimizer, please use either DNN or FM '
        'model.')


def _fm_learning_rate(num_fm_feature_columns):
  """Returns the default learning rate of the fm model.

  Args:
    num_fm_feature_columns: The number of feature columns of the fm model.

  Returns:
    A float.
  """
  default_learning_rate = 1. / math.sqrt(num_fm_feature_columns)
  return min(_FM_LEARNING_RATE, default_learning_rate)


def _add_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)


def _deepfm_model_fn(features,
                     labels,
                     mode,
                     head,
                     fm_first_feature_columns=None,
                     fm_second_feature_columns=None,
                     embedding_size=None,
                     field_size=None,
                     linear_optimizer='Ftrl',
                     dnn_feature_columns=None,
                     dnn_optimizer='Adagrad',
                     dnn_hidden_units=None,
                     dnn_activation_fn=nn.relu,
                     dnn_dropout=None,
                     input_layer_partitioner=None,
                     config=None):
  """DNN and FM combined model_fn.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
      `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `Head` instance.
    fm_first_feature_columns: An iterable containing order-1 feature columns used
      by the fm model.
    fm_second_feature_columns: An iterable containing order-2 feature columns used
      by the fm model.
    embedding_size: input field vectors can be of different sizes, their embeddings are
      of the same size.
    field_size: The number of order-2 feature columns.
    linear_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the FM model. Defaults to the Ftrl
      optimizer.
    dnn_feature_columns: An iterable containing all the feature columns used by
      the DNN model.
    dnn_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the DNN model. Defaults to the Adagrad
      optimizer.
    dnn_hidden_units: List of hidden units per DNN layer.
    dnn_activation_fn: Activation function applied to each DNN layer. If `None`,
      will use `tf.nn.relu`.
    dnn_dropout: When not `None`, the probability we will drop out a given DNN
      coordinate.
    input_layer_partitioner: Partitioner for input layer.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If both `fm_first_feature_columns` and `fm_second_feature_columns`
      and `dnn_features_columns` are em pty at the same time, or `input_layer_partitioner`
      is missing, or features has the wrong type.
  """
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))
  if not fm_first_feature_columns and not dnn_feature_columns and not fm_second_feature_columns:
    raise ValueError(
        'Either fm_first_feature_columns or dnn_feature_columns or fm_second_feature_columns must be defined.')

  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = input_layer_partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  # Build DNN Logits.
  dnn_parent_scope = 'dnn'

  if not dnn_feature_columns:
    dnn_logits = None
  else:
    dnn_optimizer = optimizers.get_optimizer_instance(
        dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
    _check_no_sync_replicas_optimizer(dnn_optimizer)
    if not dnn_hidden_units:
      raise ValueError(
          'dnn_hidden_units must be defined when dnn_feature_columns is '
          'specified.')
    dnn_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas))
    with variable_scope.variable_scope(
        dnn_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=dnn_partitioner):

      dnn_logit_fn = dnn._dnn_logit_fn_builder(
          units=head.logits_dimension,
          hidden_units=dnn_hidden_units,
          feature_columns=dnn_feature_columns,
          activation_fn=dnn_activation_fn,
          dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner)
      dnn_logits = dnn_logit_fn(features=features, mode=mode)

  # Build FM Logits.
  fm_parent_scope = 'fm'

  def cal_fm_first_logits():
      logit_fn = linear._linear_logit_fn_builder(units=head.logits_dimension, feature_columns=fm_first_feature_columns)
      fm_first_logits = logit_fn(features=features)
      _add_layer_summary(fm_first_logits, scope.name)
      return fm_first_logits

  def cal_fm_second_logits():
      embeddings = tf.feature_column.input_layer(features=features, feature_columns=fm_second_feature_columns)
      embeddings = tf.reshape(embeddings, shape=[-1, field_size, embedding_size])
      sum_square = tf.square(tf.reduce_sum(embeddings, 1))
      square_sum = tf.reduce_sum(tf.square(embeddings), 1)
      fm_second_logits = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1, keep_dims=True)
      _add_layer_summary(fm_second_logits, scope.name)
      return fm_second_logits

  if not fm_first_feature_columns and not fm_second_feature_columns:
    fm_first_logits = None
    fm_second_logits = None
  else:
      linear_optimizer = optimizers.get_optimizer_instance(linear_optimizer, learning_rate=_fm_learning_rate(len(fm_first_feature_columns) + len(fm_second_feature_columns)))
      _check_no_sync_replicas_optimizer(linear_optimizer)
      with variable_scope.variable_scope(fm_parent_scope, values=tuple(six.itervalues(features)), partitioner=input_layer_partitioner) as scope:
        if not fm_first_feature_columns:
          fm_first_logits = None
          fm_second_logits = cal_fm_second_logits()
        elif not fm_second_feature_columns:
          fm_second_logits = None
          fm_first_logits = cal_fm_first_logits()
        else:
          fm_first_logits = cal_fm_first_logits()
          fm_second_logits = cal_fm_second_logits()

  def add_logits(logits, to_add_logits):
    if logits is None:
      return to_add_logits
    else:
      return logits + to_add_logits if to_add_logits is not None else logits

  # Combine logits and build full model.
  logits = None
  logits = add_logits(logits, dnn_logits)
  logits = add_logits(logits, fm_second_logits)
  logits = add_logits(logits, fm_first_logits)


  def _train_op_fn(loss):
    """Returns the op to optimize the loss."""
    train_ops = []
    global_step = training_util.get_global_step()
    if dnn_logits is not None:
      train_ops.append(
          dnn_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=dnn_parent_scope)))
    if fm_first_logits is not None or fm_second_logits is not None:
      train_ops.append(
          linear_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=fm_parent_scope)))
    train_op = control_flow_ops.group(*train_ops)
    with ops.control_dependencies([train_op]):
      return distribute_lib.increment_var(global_step)

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)

class DeepFM(estimator.Estimator):
  def __init__(self,
               model_dir=None,
               fm_first_feature_columns=None,
               fm_second_feature_columns=None,
               linear_optimizer='Ftrl',
               dnn_feature_columns=None,
               dnn_optimizer='Adagrad',
               dnn_hidden_units=None,
               embedding_size=None,
               field_size=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               input_layer_partitioner=None,
               config=None,
               warm_start_from=None,
               loss_reduction=losses.Reduction.SUM):
    fm_first_feature_columns = fm_first_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    fm_second_feature_columns = fm_second_feature_columns or []
    self._feature_columns = (
        list(fm_first_feature_columns) + list(dnn_feature_columns) + list(fm_second_feature_columns))
    if not self._feature_columns:
      raise ValueError('Either fm_first_feature_columns or fm_second_feature_columns or dnn_feature_columns '
                       'must be defined.')
    if n_classes == 2:
      head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)
    else:
      head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
          n_classes,
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)

    def _model_fn(features, labels, mode, config):
      """Call the _deepfm_model_fn."""
      return _deepfm_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          fm_first_feature_columns=fm_first_feature_columns,
          fm_second_feature_columns=fm_second_feature_columns,
          embedding_size=embedding_size,
          field_size=field_size,
          linear_optimizer=linear_optimizer,
          dnn_feature_columns=dnn_feature_columns,
          dnn_optimizer=dnn_optimizer,
          dnn_hidden_units=dnn_hidden_units,
          dnn_activation_fn=dnn_activation_fn,
          dnn_dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config)

    super(DeepFM, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config,
        warm_start_from=warm_start_from)
