import tensorflow as tf
import numpy as np
is_on_tpu = False

def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4. "
                     "Shape: " + str(static_shape))
  # Add support for left padding.
  if kwargs.get("padding") == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    # Set middle two dimensions to None to prevent convolution from complaining
    inputs.set_shape([static_shape[0], None, None, static_shape[3]])
    kwargs["padding"] = "VALID"

def conv(inputs, filters, kernel_size, dilation_rate=(1, 1), **kwargs):
  return conv_internal(
      tf.layers.conv2d,
      inputs,
      filters,
      kernel_size,
      dilation_rate=dilation_rate,
      **kwargs)

def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
  return tf.squeeze(
      conv(
          tf.expand_dims(inputs, 2),
          filters, (kernel_size, 1),
          dilation_rate=(dilation_rate, 1),
          **kwargs), 2)

def dense(x, units, **kwargs):
  #"""Identical to tf.layers.dense, Memory optimization on tpu."""
  #fn = lambda x: tf.layers.dense(x, units, **kwargs)
  #if is_on_tpu():
  #  # TODO(noam): remove this hack once XLA does the right thing.
  #  # Forces the gradients on the inputs to be computed before the variables
  #  # are updated.  This saves memory by preventing XLA from making an extra
  #  # copy of the variables.
  #  return _recompute_grad(fn, [x])
  #else:
  #  return fn(x)
  return tf.layers.dense(x, units)

def layer_norm_compute_python(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias



def layer_norm_compute(x, epsilon, scale, bias):
  return layer_norm_compute_python(x, epsilon, scale, bias)


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "layer_norm_scale", [filters], initializer=tf.ones_initializer())
    #scale = tf.Variable(
    #    name="layer_norm_scale", initial_value=tf.ones(shape=[filters]))
    #bias = tf.Variable(
    #    name="layer_norm_bias", initial_value=tf.zeros(shape=[filters]))
    bias = tf.get_variable(
        "layer_norm_bias", [filters], initializer=tf.zeros_initializer())

    result = layer_norm_compute_python(x, epsilon, scale, bias)
    return result


def group_norm(x, filters=None, num_groups=8, epsilon=1e-5):
  """Group normalization as in https://arxiv.org/abs/1803.08494."""
  x_shape = shape_list(x)
  if filters is None:
    filters = x_shape[-1]
  assert len(x_shape) == 4
  assert filters % num_groups == 0
  # Prepare variables.
  scale = tf.get_variable(
      "group_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "group_norm_bias", [filters], initializer=tf.zeros_initializer())
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  # Reshape and compute group norm.
  x = tf.reshape(x, x_shape[:-1] + [num_groups, filters // num_groups])
  # Calculate mean and variance on heights, width, channels (not groups).
  mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return tf.reshape(norm_x, x_shape) * scale + bias


def noam_norm(x, epsilon=1.0, name=None):
  """One version of layer normalization."""
  with tf.name_scope(name, default_name="noam_norm", values=[x]):
    shape = x.get_shape()
    ndims = len(shape)
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
        tf.to_float(shape[-1])))


def apply_norm(x, norm_type, depth, epsilon):
  """Apply Normalization."""
  if norm_type == "layer":
    return layer_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "group":
    return group_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "batch":
    return tf.layers.batch_normalization(x, epsilon=epsilon)
  if norm_type == "noam":
    return noam_norm(x, epsilon)
  if norm_type == "none":
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'none'.")

def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None):
  """Apply a sequence of functions to the input or output of a layer."""

  # 对某一层的输出数据进行或前或后处理

  with tf.variable_scope(name, default_name=default_name):
    if sequence == "none":
      return x
    for c in sequence:
      if c == "a":
        x += previous_value
      elif c == "n":
        x = apply_norm(x, norm_type, depth, epsilon)
      else:
        assert c == "d", ("Unknown sequence step %s" % c)
        x = dropout_with_broadcast_dims(
            x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    return x


def layer_preprocess(layer_input, hparams):
  # 预处理，添加值，或进行normalization，或dropout

  assert "a" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence=hparams.layer_preprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_type=hparams.norm_type,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_prepostprocess")


def layer_postprocess(layer_input, layer_output, hparams):
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=hparams.layer_postprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_type=hparams.norm_type,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_postprocess")


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'",
                       x.name, x.device, cast_x.device)
  return cast_x


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
  assert "noise_shape" not in kwargs
  if broadcast_dims:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    # Allow dimensions like "-1" as well.
    broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in range(ndims)]
  return tf.nn.dropout(x, keep_prob, **kwargs)

def comma_separated_string_to_integer_list(s):
  return [int(i) for i in s.split(",") if i]

def should_generate_summaries():
  """Is this an appropriate context to generate summaries.

  Returns:
    a boolean
  """
  if "while/" in tf.contrib.framework.get_name_scope():
    # Summaries don't work well within tf.while_loop()
    return False
  if tf.get_variable_scope().reuse:
    # Avoid generating separate summaries for different data shards
    return False
  return True

def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  # tranformer中使用：length,length,-1,0, out_shape=[1,1,length,length]
  # 使用numpy构建下三角矩阵
  """Matrix band part of ones."""
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.matrix_band_part(tf.ones([rows, cols]),
                               tf.cast(num_lower, tf.int64),
                               tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band

def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


def reshape_like(a, b):
  """Reshapes a to match the shape of b in all but the last dimension."""
  ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
  if not tf.contrib.eager.in_eager_mode():
    ret.set_shape(b.get_shape().as_list()[:-1] + a.get_shape().as_list()[-1:])
  return ret

def approximate_split(x, num_splits, axis=0):
  """Split approximately equally into num_splits parts.

  Args:
    x: a Tensor
    num_splits: an integer
    axis: an integer.

  Returns:
    a list of num_splits Tensors.
  """
  size = shape_list(x)[axis]
  size_splits = [tf.div(size + i, num_splits) for i in range(num_splits)]
  return tf.split(x, size_splits, axis=axis)

def reshape_like_all_dims(a, b):
  """Reshapes a to match the shape of b."""
  ret = tf.reshape(a, tf.shape(b))
  if not tf.contrib.eager.in_eager_mode():
    ret.set_shape(b.get_shape())
  return ret

def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # 两个全连接层，inputs -> filter -> outputs
  layer_name = "%s_{}" % name if name else "{}"
  h = dense(
      inputs,
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      name=layer_name.format("conv1"))

  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      name=layer_name.format("conv2"))
  return o


def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None,
                   cache=None):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope(name, "conv_relu_conv", [inputs]):
    inputs = maybe_zero_out_padding(
        inputs, first_kernel_size, nonpadding_mask)

    if cache:
      inputs = cache["f"] = tf.concat([cache["f"], inputs], axis=1)
      inputs = cache["f"] = inputs[:, -first_kernel_size:, :]

    h = tpu_conv1d(inputs, filter_size, first_kernel_size, padding=padding,
                   name="conv1")

    if cache:
      h = h[:, -1:, :]

    h = tf.nn.relu(h)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    return tpu_conv1d(h, output_size, second_kernel_size, padding=padding,
                      name="conv2")


def maybe_zero_out_padding(inputs, kernel_size, nonpadding_mask):
  """If necessary, zero out inputs to a conv for padding positions.

  Args:
    inputs: a Tensor with shape [batch, length, ...]
    kernel_size: an integer or pair of integers
    nonpadding_mask: a Tensor with shape [batch, length]

  Returns:
    a Tensor with the same shape as inputs
  """
  if (kernel_size != 1 and
      kernel_size != (1, 1) and
      nonpadding_mask is not None):
    while nonpadding_mask.get_shape().ndims < inputs.get_shape().ndims:
      nonpadding_mask = tf.expand_dims(nonpadding_mask, -1)
    return inputs * nonpadding_mask

  return inputs

def tpu_conv1d(inputs, filters, kernel_size, padding="SAME", name="tpu_conv1d"):
  """Version of conv1d that works on TPU (as of 11/2017).

  Args:
    inputs: a Tensor with shape [batch, length, input_depth].
    filters: an integer.
    kernel_size: an integer.
    padding: a string - "SAME" or "LEFT".
    name: a string.

  Returns:
    a Tensor with shape [batch, length, filters].
  """
  if kernel_size == 1:
    return dense(inputs, filters, name=name, use_bias=True)
  if padding == "SAME":
    assert kernel_size % 2 == 1
    first_offset = -((kernel_size - 1) // 2)
  else:
    assert padding == "LEFT"
    first_offset = -(kernel_size - 1)
  last_offset = first_offset + kernel_size - 1
  results = []
  padded = tf.pad(inputs, [[0, 0], [-first_offset, last_offset], [0, 0]])
  for i in range(kernel_size):
    shifted = tf.slice(padded, [0, i, 0], tf.shape(inputs)) if i else inputs
    shifted.set_shape(inputs.get_shape())
    results.append(dense(
        shifted, filters, use_bias=(i == 0), name=name + "_%d" % i))
  ret = tf.add_n(results)
  ret *= kernel_size ** -0.5
  return ret


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     dropout=0.0,
                     **kwargs):
  """Hidden layer with RELU activation followed by linear projection."""
  name = kwargs.pop("name") if "name" in kwargs else None
  with tf.variable_scope(name, "conv_hidden_relu", [inputs]):
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    conv_f1 = conv if kernel_size == (1, 1) else separable_conv
    h = conv_f1(
        inputs,
        hidden_size,
        kernel_size,
        activation=tf.nn.relu,
        name="conv1",
        **kwargs)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    conv_f2 = conv if second_kernel_size == (1, 1) else separable_conv
    ret = conv_f2(h, output_size, second_kernel_size, name="conv2", **kwargs)
    if is_3d:
      ret = tf.squeeze(ret, 2)
    return ret

def separable_conv(inputs, filters, kernel_size, **kwargs):
  return conv_internal(tf.layers.separable_conv2d, inputs, filters, kernel_size,
                       **kwargs)


def sru(x, num_layers=2,
      activation=None, initial_state=None, name=None, reuse=None):
  """SRU cell as in https://arxiv.org/abs/1709.02755.

  As defined in the paper:
  (1) x'_t = W x_t
  (2) f_t = sigmoid(Wf x_t + bf)
  (3) r_t = sigmoid(Wr x_t + br)
  (4) c_t = f_t * c_{t-1} + (1 - f_t) * x'_t
  (5) h_t = r_t * activation(c_t) + (1 - r_t) * x_t

  This version uses functional ops to be faster on GPUs with TF-1.9+.

  Args:
    x: A tensor of shape [batch, ..., channels] ; ... is treated as time.
    num_layers: How many SRU layers; default is 2 as results for 1 disappoint.
    activation: Optional activation function, try tf.nn.tanh or tf.nn.relu.
    initial_state: Optional initial c-state, set to zeros if None.
    name: Optional name, "sru" by default.
    reuse: Optional reuse.

  Returns:
    A tensor of the same shape as x.

  Raises:
    ValueError: if num_layers is not positive.
  """
  if num_layers < 1:
      raise ValueError("Number of layers must be positive: %d" % num_layers)
  if is_on_tpu():  # On TPU the XLA does a good job with while.
      return sru_with_scan(x, num_layers, activation, initial_state, name, reuse)
  try:
      from tensorflow.contrib.recurrent.python.ops import functional_rnn  # pylint: disable=g-import-not-at-top
  except ImportError:
      tf.logging.info("functional_rnn not found, using sru_with_scan instead")
      return sru_with_scan(x, num_layers, activation, initial_state, name, reuse)

  with tf.variable_scope(name, default_name="sru", values=[x], reuse=reuse):
      # We assume x is [batch, ..., channels] and treat all ... as time.
      x_shape = shape_list(x)
      x = tf.reshape(x, [x_shape[0], -1, x_shape[-1]])
      initial_state = initial_state or tf.zeros([x_shape[0], x_shape[-1]])
      cell = CumsumprodCell(initial_state)
      # Calculate SRU on each layer.
      for i in range(num_layers):
          # The parallel part of the SRU.
          x_orig = x
          x, f, r = tf.split(tf.layers.dense(x, 3 * x_shape[-1],
                                             name="kernel_%d" % i), 3, axis=-1)
          f, r = tf.sigmoid(f), tf.sigmoid(r)
          x_times_one_minus_f = x * (1.0 - f)  # Compute in parallel for speed.
          # Calculate states.
          concat = tf.concat([x_times_one_minus_f, f], axis=-1)
          c_states, _ = functional_rnn.functional_rnn(
              cell, concat, time_major=False)
          # Final output.
          if activation is not None:
              c_states = activation(c_states)
          h = c_states * r + (1.0 - r) * x_orig
          x = h  # Next layer.
      return tf.reshape(x, x_shape)

def sru_with_scan(x, num_layers=2,
                  activation=None, initial_state=None, name=None, reuse=None):
  """SRU cell as in https://arxiv.org/abs/1709.02755.

  This implementation uses tf.scan and can incur overhead, see the full SRU
  function doc for details and an implementation that is sometimes faster.

  Args:
    x: A tensor of shape [batch, ..., channels] ; ... is treated as time.
    num_layers: How many SRU layers; default is 2 as results for 1 disappoint.
    activation: Optional activation function, try tf.nn.tanh or tf.nn.relu.
    initial_state: Optional initial c-state, set to zeros if None.
    name: Optional name, "sru" by default.
    reuse: Optional reuse.

  Returns:
    A tensor of the same shape as x.

  Raises:
    ValueError: if num_layers is not positive.
  """
  if num_layers < 1:
    raise ValueError("Number of layers must be positive: %d" % num_layers)
  with tf.variable_scope(name, default_name="sru", values=[x], reuse=reuse):
    # We assume x is [batch, ..., channels] and treat all ... as time.
    x_shape = shape_list(x)
    x = tf.reshape(x, [x_shape[0], -1, x_shape[-1]])
    x = tf.transpose(x, [1, 0, 2])  # Scan assumes time on axis 0.
    initial_state = initial_state or tf.zeros([x_shape[0], x_shape[-1]])
    # SRU state manipulation function.
    def next_state(cur_state, args_tup):
      cur_x_times_one_minus_f, cur_f = args_tup
      return cur_f * cur_state + cur_x_times_one_minus_f
    # Calculate SRU on each layer.
    for i in range(num_layers):
      # The parallel part of the SRU.
      x_orig = x
      x, f, r = tf.split(tf.layers.dense(x, 3 * x_shape[-1],
                                         name="kernel_%d" % i), 3, axis=-1)
      f, r = tf.sigmoid(f), tf.sigmoid(r)
      x_times_one_minus_f = x * (1.0 - f)  # Compute in parallel for speed.
      # Calculate states.
      c_states = tf.scan(next_state, (x_times_one_minus_f, f),
                         initializer=initial_state,
                         parallel_iterations=2, name="scan_%d" % i)
      # Final output.
      if activation is not None:
        c_states = activation(c_states)
      h = c_states * r + (1.0 - r) * x_orig
      x = h  # Next layer.
    # Transpose back to batch-major.
    x = tf.transpose(x, [1, 0, 2])
    return tf.reshape(x, x_shape)

class CumsumprodCell(object):
  """Cumulative sum and product object for use with functional_rnn API."""

  def __init__(self, initializer):
    self._initializer = initializer

  @property
  def output_size(self):
    return int(shape_list(self._initializer)[-1])

  def zero_state(self, batch_size, dtype):
    dtype = dtype or tf.float32
    return tf.zeros([batch_size, self.output_size], dtype=dtype)

  def __call__(self, inputs_t, state_t):
    cur_x_times_one_minus_f, cur_f = tf.split(inputs_t, 2, axis=-1)
    state_next = cur_f * state_t + cur_x_times_one_minus_f
    outputs_t = state_next
    return outputs_t, state_next
