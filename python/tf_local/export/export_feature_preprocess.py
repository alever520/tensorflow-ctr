import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.framework import dtypes


def build_parsing_serving_input_receiver_fn(feature_spec, input_raw_feature_params=None,
                                            default_batch_size=None):
  """Build a serving_input_receiver_fn expecting fed tf.Examples.

  Creates a serving_input_receiver_fn that expects a serialized tf.Example fed
  into a string placeholder.  The function parses the tf.Example according to
  the provided feature_spec, and returns all parsed Tensors as features.

  Args:
    feature_spec: a dict of string to `VarLenFeature`/`FixedLenFeature`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    A serving_input_receiver_fn suitable for use in serving.
  """

  def serving_input_receiver_fn():
    """An input_fn that expects a serialized tf.Example."""
    serialized_tf_example = array_ops.placeholder(dtype=dtypes.string,
                                                  shape=[default_batch_size],
                                                  name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = parsing_ops.parse_example(serialized_tf_example, feature_spec)
    print(features)
    if input_raw_feature_params:
      for elem in input_raw_feature_params:
        if elem["used"] != "true" or not elem.get("is_serving_feature_preprocess", False):
          continue
        feature = elem["name"]
        # not work, because of shape[0] is None before graph
        # if features[feature].shape[0] < 2:
        #   continue
        features[feature] = tf.tile([features[feature][0]], tf.concat([[tf.shape(features[feature])[0]], tf.tile([1], [tf.rank(features[feature])-1])], axis=0))

    print(features)
    for feature, tensor in features.iteritems():
      features[feature] = tf.Print(tensor, [tensor])

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  return serving_input_receiver_fn
