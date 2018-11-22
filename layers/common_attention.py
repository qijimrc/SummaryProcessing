import tensorflow as tf
from layers import common_layers
import math

def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = common_layers.shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])

def split_heads(x, num_heads):
  # 将数据在最后一个维度分割，便于并行
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])

def attention_bias_to_padding(attention_bias):

  return tf.squeeze(tf.to_float(tf.less(attention_bias, -1)), axis=[1, 2])


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID"):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: and integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  # 预处理q, k, v
  # 对每一个最后一个维度的向量做卷积或全连接
  # 相当于逐字处理

  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  def _compute(inp, depth, filter_width, padding, name):
    if filter_width == 1:
      return tf.layers.dense(inp, depth, use_bias=False, name=name)
    return common_layers.conv1d(inp, depth, filter_width, padding, name=name)
  q = _compute(
      query_antecedent, total_key_depth, q_filter_width, q_padding, "q")
  k = _compute(
      memory_antecedent, total_key_depth, kv_filter_width, kv_padding, "k")
  v = _compute(
      memory_antecedent, total_value_depth, kv_filter_width, kv_padding, "v")
  return q, k, v

def attention_image_summary(attn, image_shapes=None):
  """Compute color image summary.

  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
  """
  attn = tf.cast(attn, tf.float32)
  num_heads = common_layers.shape_list(attn)[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.mod(-num_heads, 3)]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  if image_shapes is not None:
    if len(image_shapes) == 4:
      q_rows, q_cols, m_rows, m_cols = list(image_shapes)
      image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
      image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
      image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
    else:
      assert len(image_shapes) == 6
      q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(
          image_shapes)
      image = tf.reshape(
          image,
          [-1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3])
      image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
      image = tf.reshape(
          image,
          [-1, q_rows * m_rows * q_channnels, q_cols * m_cols * m_channels, 3])
  tf.summary.image("attention", image, max_outputs=1)

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None):
  # 做点积attention通用方法
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      # 加负无穷mask
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # dropping out the attention links for each of the heads
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


# 核心方法，Multihed-Attention
def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        max_relative_position=None,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        max_length=None,
                        **kwargs):
  # multihead的实际做法是：
  # 1. 先将数据分割再转置：[batch, seq_len,hidden_dim]->[batch,num_heads,seq_len,hidden_dim//num_heads]
  # 2. 然后做scaled-dot attention
  # 3. 然后再合并转置

  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):
    # 对每一个word vector进行变换(全连接)
    q, k, v = compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                          total_value_depth, q_filter_width, kv_filter_width,
                          q_padding, kv_padding)

    if cache is not None:
      if attention_type != "dot_product":
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")
      k = cache["k"] = tf.concat([cache["k"], k], axis=1)
      v = cache["v"] = tf.concat([cache["v"], v], axis=1)

    # 在最后一个维度分割，并转置。[batch, num_heads, length, channels/num_heads]
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    # scaled 缩放步骤
    q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                                save_weights_to=save_weights_to,
                                make_image_summary=make_image_summary,
                                dropout_broadcast_dims=dropout_broadcast_dims)
    elif attention_type == "dot_product_relative":
      x = dot_product_attention_relative(q, k, v, bias, max_relative_position,
                                         dropout_rate, image_shapes,
                                         make_image_summary=make_image_summary)
    elif attention_type == "dot_product_relative_v2":
      x = dot_product_self_attention_relative_v2(
          q, k, v, bias, max_length, dropout_rate, image_shapes,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims)
    elif attention_type == "local_within_block_mask_right":
      x = masked_within_block_local_attention_1d(q, k, v,
                                                 block_length=block_length)
    elif attention_type == "local_mask_right":
      x = masked_local_attention_1d(q, k, v, block_length=block_length,
                                    make_image_summary=make_image_summary)
    elif attention_type == "local_unmasked":
      x = local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                           gap_size, num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = dilated_self_attention_1d(q, k, v, block_length, block_width,
                                    gap_size, num_memory_blocks)
    # 由multihead转回原来的样子[batch, length, channels]
    x = combine_heads(x)
    # 再将每个词对应向量分别过全连接
    x = common_layers.dense(
        x, output_depth, use_bias=False, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x

def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   dropout_rate=0.0,
                                   image_shapes=None,
                                   name=None,
                                   make_image_summary=True):
  """Calculate relative position-aware dot-product self-attention.

  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.

  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer specifying the maximum distance between
        inputs that unique position embeddings should be learned for.
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.

  Returns:
    A Tensor.

  Raises:
    ValueError: if max_relative_position is not > 0.
  """
  if not max_relative_position:
    raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))
  with tf.variable_scope(
      name, default_name="dot_product_attention_relative", values=[q, k, v]):

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    depth = q.get_shape().as_list()[3]
    length = common_layers.shape_list(q)[2]
    relations_keys = _generate_relative_positions_embeddings(
        length, depth, max_relative_position, "relative_positions_keys")
    relations_values = _generate_relative_positions_embeddings(
        length, depth, max_relative_position, "relative_positions_values")

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(q, k, relations_keys, True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if not tf.get_variable_scope().reuse and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return _relative_attention_inner(weights, v, relations_values, False)

def dot_product_self_attention_relative_v2(q,
                                           k,
                                           v,
                                           bias,
                                           max_length=None,
                                           dropout_rate=0.0,
                                           image_shapes=None,
                                           name=None,
                                           make_image_summary=True,
                                           dropout_broadcast_dims=None):
  """Calculate relative position-aware dot-product self-attention.

  Only works for masked self-attention (no looking forward).
  TODO(noam): extend to unmasked self-attention

  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.

  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_length: an integer - changing this invalidates checkpoints
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_self_attention_relative_v2",
      values=[q, k, v]):

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    length = common_layers.shape_list(q)[2]
    assert max_length is not None

    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)

    # now add relative logits
    # [batch, num_heads, query_length, max_length]
    rel_logits = common_layers.dense(q, max_length, name="rel0")
    # [batch, num_heads, query_length, max_length]
    rel_logits = tf.slice(
        rel_logits, [0, 0, 0, max_length - length], [-1, -1, -1, -1])
    rel_logits = _relative_position_to_absolute_position_masked(rel_logits)
    logits += rel_logits

    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    ret = tf.matmul(weights, v)
    # [batch, num_heads, query_length, memory_length]
    relative_weights = _absolute_position_to_relative_position_masked(weights)
    # [batch, num_heads, query_length, memory_length]
    relative_weights = tf.pad(
        relative_weights, [[0, 0], [0, 0], [0, 0], [max_length - length, 0]])
    relative_weights.set_shape([None, None, None, max_length])
    depth_v = common_layers.shape_list(v)[3]
    ret += common_layers.dense(relative_weights, depth_v, name="rel1")
    return ret

def _generate_relative_positions_embeddings(length, depth,
                                            max_relative_position, name):
  """Generates tensor of size [length, length, depth]."""
  with tf.variable_scope(name):
    relative_positions_matrix = _generate_relative_positions_matrix(
        length, max_relative_position)
    vocab_size = max_relative_position * 2 + 1
    # Generates embedding for each relative position of dimension depth.
    embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
    embeddings = tf.gather(embeddings_table, relative_positions_matrix)
    return embeddings

def masked_within_block_local_attention_1d(q, k, v, block_length=64, name=None):
  """Attention to the source and a neighborhood to the left within a block.

  The sequence is divided into blocks of length block_size.
  Attention for a given query position can only see memory positions
  less than or equal to the query position in the corresponding block

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="within_local_attention_1d", values=[q, k, v]):
    v_shape = v.get_shape()
    batch, heads, length, _ = common_layers.shape_list(q)
    if isinstance(block_length, tf.Tensor):
      const = tf.contrib.util.constant_value(block_length)
      if const is not None:
        block_length = int(const)

    depth_k = common_layers.shape_list(k)[3]
    depth_v = common_layers.shape_list(v)[3]
    original_length = length
    padding_size = tf.mod(-length, block_length)
    length += padding_size
    padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
    q = tf.pad(q, padding)
    k = tf.pad(k, padding)
    v = tf.pad(v, padding)
    num_blocks = tf.div(length, block_length)
    # compute attention for all subsequent query blocks.
    q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
    k = tf.reshape(k, [batch, heads, num_blocks, block_length, depth_k])
    v = tf.reshape(v, [batch, heads, num_blocks, block_length, depth_v])
    # attention shape: [batch, heads, num_blocks, block_length, block_length]
    attention = tf.matmul(q, k, transpose_b=True)
    attention += tf.reshape(attention_bias_lower_triangle(block_length),
                            [1, 1, 1, block_length, block_length])
    attention = tf.nn.softmax(attention)
    # initial output shape: [batch, heads, num_blocks, block_length, depth_v]
    output = tf.matmul(attention, v)
    output = tf.reshape(output, [batch, heads, -1, depth_v])
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_shape)
    return output

def masked_local_attention_1d(q, k, v, block_length=128,
                              make_image_summary=False, name=None):
  """Attention to the source position and a neighborhood to the left of it.

  The sequence is divided into blocks of length block_size.
  Attention for a given query position can only see memory positions
  less than or equal to the query position, in the corresponding block
  and the previous block.

  If mask_right is True, then a target position cannot see greater source
  positions.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    make_image_summary: a boolean, whether to make an attention image summary.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_attention_1d", values=[q, k, v]):
    batch = common_layers.shape_list(q)[0]
    heads = common_layers.shape_list(q)[1]
    length = common_layers.shape_list(q)[2]
    if isinstance(block_length, tf.Tensor):
      const = tf.contrib.util.constant_value(block_length)
      if const is not None:
        block_length = int(const)

    # If (length < 2 * block_length), then we use only one block.
    if isinstance(length, int) and isinstance(block_length, int):
      block_length = length if length < block_length * 2 else block_length
    else:
      block_length = tf.where(
          tf.less(length, block_length * 2), length, block_length)
    depth_k = common_layers.shape_list(k)[3]
    depth_v = common_layers.shape_list(v)[3]
    original_length = length
    padding_size = tf.mod(-length, block_length)
    length += padding_size
    padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
    q = tf.pad(q, padding)
    k = tf.pad(k, padding)
    v = tf.pad(v, padding)

    if isinstance(length, int) and isinstance(block_length, int):
      num_blocks = length // block_length
    else:
      num_blocks = tf.div(length, block_length)

    # compute attention for the first query block.
    first_q = tf.slice(q, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_k = tf.slice(k, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_v = tf.slice(v, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_output = dot_product_attention(
        first_q,
        first_k,
        first_v,
        attention_bias_lower_triangle(block_length),
        make_image_summary=make_image_summary,
        name="fist_block")

    # compute attention for all subsequent query blocks.
    q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
    k = tf.reshape(k, [batch, heads, num_blocks, block_length, depth_k])
    v = tf.reshape(v, [batch, heads, num_blocks, block_length, depth_v])

    def local(x, depth):
      """Create a local version of the keys or values."""
      prev_block = tf.slice(x, [0, 0, 0, 0, 0],
                            [-1, -1, num_blocks - 1, -1, -1])
      cur_block = tf.slice(x, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
      local_block = tf.concat([prev_block, cur_block], 3)
      return tf.reshape(local_block,
                        [batch, heads, num_blocks - 1,
                         block_length * 2, depth])

    local_k = local(k, depth_k)
    local_v = local(v, depth_v)
    tail_q = tf.slice(q, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
    tail_q = tf.reshape(tail_q, [batch, heads, num_blocks - 1,
                                 block_length, depth_k])
    local_length = common_layers.shape_list(local_k)[3]

    # [batch, heads, num_blocks - 1, block_length, local_length]
    attention = tf.matmul(tail_q, local_k, transpose_b=True)

    # make sure source_pos <= target_pos
    good_part = common_layers.ones_matrix_band_part(block_length, local_length,
                                                    -1, block_length)
    mask = (1.0 - good_part) * -1e9
    mask = common_layers.cast_like(mask, attention)
    attention += tf.reshape(mask, [1, 1, 1, block_length, local_length])
    attention = tf.nn.softmax(attention)
    # TODO(noam): figure out how to show a summary for the remaining blocks.
    # The naive way currently causes errors due to empty tensors.
    # output: [batch, heads, num_blocks-1, block_length, depth_v]
    output = tf.matmul(attention, local_v)
    output = tf.reshape(output, [
        batch, heads, (num_blocks-1)*block_length, depth_v])
    output = tf.concat([first_output, output], axis=2)
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output = tf.reshape(output, [batch, heads, original_length, depth_v])
    return output

def local_attention_1d(q, k, v, block_length=128, filter_width=100, name=None):
  """strided block local self-attention.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    filter_width: an integer indicating how much to look left.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_self_attention_1d", values=[q, k, v]):
    v_shape = v.get_shape()
    depth_v = common_layers.shape_list(v)[3]
    batch_size = common_layers.shape_list(q)[0]
    num_heads = common_layers.shape_list(q)[1]
    original_length = common_layers.shape_list(q)[2]

    # making sure q is a multiple of d
    def pad_to_multiple(x, pad_length):
      x_length = common_layers.shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    q = pad_to_multiple(q, block_length)
    k = pad_to_multiple(k, block_length)
    v = pad_to_multiple(v, block_length)

    # Setting up q blocks
    new_q_shape = common_layers.shape_list(q)
    # Setting up q blocks
    q = tf.reshape(q, [
        new_q_shape[0], new_q_shape[1], new_q_shape[2] // block_length,
        block_length, new_q_shape[3]
    ])

    # Setting up k and v values
    k = pad_l_and_r(k, filter_width)
    v = pad_l_and_r(v, filter_width)

    length = common_layers.shape_list(k)[2]
    full_filter_width = block_length + 2 * filter_width
    # getting gather indices
    indices = tf.range(0, length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(full_filter_width), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        block_length,
        padding="VALID",
        name="gather_conv")

    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    k_new = tf.gather(k_t, gather_indices)

    # [batch, heads, blocks, block_length, dim]
    k_new = tf.transpose(k_new, [2, 3, 0, 1, 4])

    attention_bias = tf.expand_dims(embedding_to_padding(k_new) * -1e9, axis=-2)

    v_t = tf.transpose(v, [2, 0, 1, 3])
    v_new = tf.gather(v_t, gather_indices)
    v_new = tf.transpose(v_new, [2, 3, 0, 1, 4])

    output = dot_product_attention(
        q,
        k_new,
        v_new,
        attention_bias,
        dropout_rate=0.,
        name="local_1d",
        make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_shape)
    return output

def masked_dilated_self_attention_1d(q,
                                     k,
                                     v,
                                     query_block_size=64,
                                     memory_block_size=64,
                                     gap_size=2,
                                     num_memory_blocks=2,
                                     name=None):
  """dilated self-attention. TODO(avaswani): Try it and write a paper on it.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    query_block_size: an integer
    memory_block_size: an integer indicating how much to look left.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left. Each will
      be separated by gap_size.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="masked_dilated_self_attention_1d", values=[q, k, v]):
    v_list_shape = v.get_shape().as_list()
    v_shape = common_layers.shape_list(v)
    depth_v = v_shape[3]
    batch_size = v_shape[0]
    num_heads = v_shape[1]
    original_length = common_layers.shape_list(q)[2]

    # making sure q is a multiple of query block size
    def pad_to_multiple(x, pad_length):
      x_length = common_layers.shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l(x, left_pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [left_pad_length, 0], [0, 0]])

    q = pad_to_multiple(q, query_block_size)
    v = pad_to_multiple(v, query_block_size)
    k = pad_to_multiple(k, query_block_size)
    q.set_shape(v_list_shape)
    v.set_shape(v_list_shape)
    k.set_shape(v_list_shape)
    # Setting up q blocks
    new_q_shape = common_layers.shape_list(q)

    # Setting up q blocks
    q = reshape_by_blocks(q, new_q_shape, query_block_size)
    self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
    self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)
    # Setting up k and v windows
    k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
    k = pad_l(k, k_v_padding)
    v = pad_l(v, k_v_padding)
    # getting gather indices
    index_length = (new_q_shape[2] - query_block_size + memory_block_size)

    indices = tf.range(0, index_length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        query_block_size,
        padding="VALID",
        name="gather_conv")
    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # get left and right memory blocks for each query
    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    v_t = tf.transpose(v, [2, 0, 1, 3])

    k_unmasked_windows = gather_dilated_memory_blocks(
        k_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
        gather_indices)
    v_unmasked_windows = gather_dilated_memory_blocks(
        v_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
        gather_indices)

    # combine memory windows
    block_q_shape = common_layers.shape_list(q)
    masked_attention_bias = tf.tile(
        tf.expand_dims(attention_bias_lower_triangle(query_block_size), axis=0),
        [block_q_shape[0], block_q_shape[1], block_q_shape[2], 1, 1])
    padding_attention_bias = tf.expand_dims(
        embedding_to_padding(k_unmasked_windows) * -1e9, axis=-2)
    padding_attention_bias = tf.tile(padding_attention_bias,
                                     [1, 1, 1, query_block_size, 1])
    attention_bias = tf.concat(
        [masked_attention_bias, padding_attention_bias], axis=-1)
    # combine memory windows
    k_windows = tf.concat([self_k_part, k_unmasked_windows], 3)
    v_windows = tf.concat([self_v_part, v_unmasked_windows], 3)
    output = dot_product_attention(
        q,
        k_windows,
        v_windows,
        attention_bias,
        dropout_rate=0.,
        name="dilated_1d",
        make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_list_shape)
    return output

def dilated_self_attention_1d(q,
                              k,
                              v,
                              query_block_size=128,
                              memory_block_size=128,
                              gap_size=2,
                              num_memory_blocks=2,
                              name=None):
  """dilated self-attention.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    query_block_size: an integer indicating size of query block
    memory_block_size: an integer indicating the size of a memory block.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left and right.
      Each will be separated by gap_size.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="dilated_self_attention_1d", values=[q, k, v]):
    v_list_shape = v.get_shape().as_list()
    v_shape = common_layers.shape_list(v)
    depth_v = v_shape[3]
    batch_size = v_shape[0]
    num_heads = v_shape[1]
    original_length = common_layers.shape_list(q)[2]

    # making sure q is a multiple of query block size
    def pad_to_multiple(x, pad_length):
      x_length = common_layers.shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    q = pad_to_multiple(q, query_block_size)
    v = pad_to_multiple(v, query_block_size)
    k = pad_to_multiple(k, query_block_size)

    q.set_shape(v_list_shape)
    v.set_shape(v_list_shape)
    k.set_shape(v_list_shape)
    # Setting up q blocks
    new_q_shape = common_layers.shape_list(q)
    # Setting up q blocks
    q = reshape_by_blocks(q, new_q_shape, query_block_size)
    self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
    self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)

    # Setting up k and v windows
    k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
    k = pad_l_and_r(k, k_v_padding)
    v = pad_l_and_r(v, k_v_padding)
    # getting gather indices
    index_length = (new_q_shape[2] - query_block_size + memory_block_size)
    indices = tf.range(0, index_length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        query_block_size,
        padding="VALID",
        name="gather_conv")

    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # get left and right memory blocks for each query
    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    v_t = tf.transpose(v, [2, 0, 1, 3])
    left_k = gather_dilated_memory_blocks(
        k_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
        query_block_size, memory_block_size, gather_indices)
    left_v = gather_dilated_memory_blocks(
        v_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
        query_block_size, memory_block_size, gather_indices)

    right_k = gather_dilated_memory_blocks(
        k_t[k_v_padding:, :, :, :],
        num_memory_blocks,
        gap_size,
        query_block_size,
        memory_block_size,
        gather_indices,
        direction="right")
    right_v = gather_dilated_memory_blocks(
        v_t[k_v_padding:, :, :, :],
        num_memory_blocks,
        gap_size,
        query_block_size,
        memory_block_size,
        gather_indices,
        direction="right")

    k_windows = tf.concat([left_k, self_k_part, right_k], axis=3)
    v_windows = tf.concat([left_v, self_v_part, right_v], axis=3)
    attention_bias = tf.expand_dims(
        embedding_to_padding(k_windows) * -1e9, axis=-2)

    output = dot_product_attention(
        q,
        k_windows,
        v_windows,
        attention_bias,
        dropout_rate=0.,
        name="dilated_1d",
        make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_list_shape)
    return output


def combine_heads(x):
  # 由multihead 转回原来的形状
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def _relative_attention_inner(x, y, z, transpose):
  """Relative position-aware dot-product attention inner calculation.

  This batches matrix multiply calculations to avoid unnecessary broadcasting.

  Args:
    x: Tensor with shape [batch_size, heads, length, length or depth].
    y: Tensor with shape [batch_size, heads, length, depth].
    z: Tensor with shape [length, length, depth].
    transpose: Whether to transpose inner matrices of y and z. Should be true if
        last dimension of x is depth, not length.

  Returns:
    A Tensor with shape [batch_size, heads, length, length or depth].
  """
  batch_size = tf.shape(x)[0]
  heads = x.get_shape().as_list()[1]
  length = tf.shape(x)[2]

  # xy_matmul is [batch_size, heads, length, length or depth]
  xy_matmul = tf.matmul(x, y, transpose_b=transpose)
  # x_t is [length, batch_size, heads, length or depth]
  x_t = tf.transpose(x, [2, 0, 1, 3])
  # x_t_r is [length, batch_size * heads, length or depth]
  x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
  # x_tz_matmul is [length, batch_size * heads, length or depth]
  x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
  # x_tz_matmul_r is [length, batch_size, heads, length or depth]
  x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
  # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
  x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
  return xy_matmul + x_tz_matmul_r_t

def _relative_position_to_absolute_position_masked(x):
  """Helper to dot_product_self_attention_relative_v2.

  Rearrange an attention logits or weights Tensor.

  The dimensions of the input represent:
  [batch, heads, query_position, memory_position - query_position + length - 1]

  The dimensions of the output represent:
  [batch, heads, query_position, memory_position]

  Only works with masked_attention.  Undefined behavior for regions of the
  input where memory_position > query_position.

  Args:
    x: a Tensor with shape [batch, heads, length, length]

  Returns:
    a Tensor with shape [batch, heads, length, length]
  """
  batch, heads, length, _ = common_layers.shape_list(x)
  x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
  x = tf.reshape(x, [batch, heads, 1 + length, length])
  x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
  return x


def _absolute_position_to_relative_position_masked(x):
  """Helper to dot_product_self_attention_relative_v2.

  Rearrange an attention logits or weights Tensor.

  The dimensions of the input represent:
  [batch, heads, query_position, memory_position]

  The dimensions of the output represent:
  [batch, heads, query_position, memory_position - query_position + length - 1]

  Only works with masked_attention.  Undefined behavior for regions of the
  input where memory_position > query_position.

  Args:
    x: a Tensor with shape [batch, heads, length, length]

  Returns:
    a Tensor with shape [batch, heads, length, length]
  """
  batch, heads, length, _ = common_layers.shape_list(x)
  x = tf.pad(x, [[0, 0], [0, 0], [1, 0], [0, 0]])
  x = tf.reshape(x, [batch, heads, length, length + 1])
  x = tf.slice(x, [0, 0, 0, 1], [batch, heads, length, length])
  return x


def _generate_relative_positions_matrix(length, max_relative_position):
  """Generates matrix of relative positions between inputs."""
  range_vec = tf.range(length)
  range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
  distance_mat = range_mat - tf.transpose(range_mat)
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
  # Shift values to be >= 0. Each integer still uniquely identifies a relative
  # position difference.
  final_mat = distance_mat_clipped + max_relative_position
  return final_mat

def attention_bias_lower_triangle(length):
  # 创建mask的bias，使得当前key只能attend前面位置的values
  # length -> seq_len
  """Create an bias tensor to be added to attention logits.

  Allows a query to attend to all positions up to and including its own.

  Args:
   length: a Scalar.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  # 返回一个下三角矩阵(上三角都是负无穷)
  return attention_bias_local(length, -1, 0)

def embedding_to_padding(emb):
  """Calculates the padding mask based on which embeddings are all zero.

  We have hacked symbol_modality to return all-zero embeddings for padding.

  Args:
    emb: a Tensor with shape [..., depth].
  Returns:
    a float Tensor with shape [...].
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.to_float(tf.equal(emb_sum, 0.0))


def reshape_by_blocks(x, x_shape, memory_block_size):
  x = tf.reshape(x, [
      x_shape[0], x_shape[1], x_shape[2] // memory_block_size,
      memory_block_size, x_shape[3]
  ])
  return x

def gather_dilated_memory_blocks(x,
                                 num_memory_blocks,
                                 gap_size,
                                 query_block_size,
                                 memory_block_size,
                                 gather_indices,
                                 direction="left"):
  """Gathers blocks with gaps in between.

  Args:
    x: A tensor of shape [length, batch, heads, depth]
    num_memory_blocks:     num_memory_blocks: how many memory blocks to look
      in "direction". Each will be separated by gap_size.
    gap_size: an integer indicating the gap size
    query_block_size: an integer indicating size of query block
    memory_block_size: an integer indicating the size of a memory block.
    gather_indices: The indices to gather from.
    direction: left or right
  Returns:
    a tensor of shape [batch, heads, blocks, block_length, depth]
  """

  gathered_blocks = []
  # gathering memory blocks
  for block_id in range(num_memory_blocks):
    block_end_index = -(query_block_size + gap_size *
                        (block_id + 1) + memory_block_size * block_id) - 1
    block_start_index = ((memory_block_size + gap_size) * (num_memory_blocks -
                                                           (block_id + 1)))
    if direction != "left":
      [block_end_index,
       block_start_index] = [-block_start_index - 1, -block_end_index + 1]

    def gather_dilated_1d_blocks(x, gather_indices):
      x_new = tf.gather(x, gather_indices)
      # [batch, heads, blocks, block_length, dim]
      return tf.transpose(x_new, [2, 3, 0, 1, 4])

    gathered_blocks.append(
        gather_dilated_1d_blocks(x[block_start_index:block_end_index],
                                 gather_indices))
  return tf.concat(gathered_blocks, 3)

def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = common_layers.shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])

def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.

  A position may attend to positions at most max_distance from it,
  forward and backwards.

  This does not actually save any computation.

  Args:
    length: int
    max_backward: int, maximum distance backward to attend. Negative values
      indicate unlimited.
    max_forward: int, maximum distance forward to attend. Negative values
      indicate unlimited.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  # 重点，怎么让每次不同位置的key看到不同长度序列
  # 使用numpy构建下三角矩阵band
  band = common_layers.ones_matrix_band_part(
      length,
      length,
      max_backward,
      max_forward,
      out_shape=[1, 1, length, length])
  return -1e9 * (1.0 - band)


def parameter_attention(x,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        memory_rows,
                        num_heads,
                        dropout_rate,
                        name=None):
  """Attention over parameters.

  We use the same multi-headed attention as in the other layers, but the memory
  keys and values are model parameters.  There are no linear transformation
  on the keys or values.

  We are also a bit more careful about memory usage, since the number of
  memory positions may be very large.

  Args:
    x: a Tensor with shape [batch, length_q, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    memory_rows: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, default_name="parameter_attention", values=[x]):
    head_size_k = total_key_depth // num_heads
    head_size_v = total_value_depth // num_heads
    var_shape_k = [num_heads, memory_rows, head_size_k]
    var_shape_v = [num_heads, memory_rows, head_size_v]
    k = tf.get_variable(
        "k",
        var_shape_k,
        initializer=tf.random_normal_initializer(0, output_depth**-0.5)) * (
            num_heads**0.5)
    v = tf.get_variable(
        "v",
        var_shape_v,
        initializer=tf.random_normal_initializer(0, output_depth**-0.5)) * (
            output_depth**0.5)
    batch_size = common_layers.shape_list(x)[0]
    length = common_layers.shape_list(x)[1]
    q = common_layers.dense(
        x, total_key_depth, use_bias=False, name="q_transform")
    if dropout_rate:
      # This is a cheaper form of attention dropout where we use to use
      # the same dropout decisions across batch elements and query positions,
      # but different decisions across heads and memory positions.
      v = tf.nn.dropout(
          v, 1.0 - dropout_rate, noise_shape=[num_heads, memory_rows, 1])
    # query is [batch, length, hidden_size]
    # reshape and transpose it to [heads, batch * length, head_size]
    q = tf.reshape(q, [batch_size, length, num_heads, head_size_k])
    q = tf.transpose(q, [2, 0, 1, 3])
    q = tf.reshape(q, [num_heads, batch_size * length, head_size_k])
    weights = tf.matmul(q, k, transpose_b=True)
    weights = tf.nn.softmax(weights)
    y = tf.matmul(weights, v)
    y = tf.reshape(y, [num_heads, batch_size, length, head_size_v])
    y = tf.transpose(y, [1, 2, 0, 3])
    y = tf.reshape(y, [batch_size, length, total_value_depth])
    y.set_shape([None, None, total_value_depth])
    y = common_layers.dense(
        y, output_depth, use_bias=False, name="output_transform")
    return y

def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.

  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].

  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = memory_padding * -1e9
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  # 给序列添加位置(时间)信号
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor the same shape as x.
  """
  length = common_layers.shape_list(x)[1]
  channels = common_layers.shape_list(x)[2]
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale,
                                start_index)
  return x + signal

def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    # 给序列生成位置（时间）信号
  """Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal
