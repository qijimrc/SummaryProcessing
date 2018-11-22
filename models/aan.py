# coding=utf-8
import tensorflow as tf
from models.base_model import base_model
from layers import common_attention
from layers import common_layers
from utils import expert_utils
import ipdb

class AAN_Model(base_model):
    def __init__(self, *args, **kwargs):
        super(AAN_Model, self).__init__(*args, **kwargs)

    def body(self,inputs):
        '''
        implements of base model
        :param inputs: [batch, seq_len, hidden_size]
        :return: [batch, seq_len, hidden_size]
        '''
        self.attention_weights = dict()  # For visualizing attention heads.
        losses = []

        # 对embedding之后的输入预处理
        # 1. 获得self_attention_bias和encoder_decoder_attention_bia，相同，padding部分减去负无穷，作为之后mask用,[None, 1, 1, seq_len]
        # 2. 添加target space的embedding到输入中
        # 3. 添加时间(位置)信号到输入中
        encoder_input, self_attention_bias = self.prepare_inputs(inputs)
        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - self.hparams.layer_prepostprocess_dropout)

        # 进行6层，每层有8头multihead+一个ffn的变换
        # 得到 [None, None, 512]
        encoder_output = self.transform_encode(
            self.hparams,
            encoder_input, self_attention_bias,
            save_weights_to=self.attention_weights,
            losses=losses)

        # 得到输出 [None, None, 512]
        return encoder_output



    def transform_encode(self,
                         hparams,
                         encoder_input,
                         encoder_self_attention_bias,
                         name="encoder",
                         save_weights_to=None,
                         losses=None):
      # 编码器

      x = encoder_input
      with tf.variable_scope(name):
        # 得到[None, seq_len], padding处为1.0, 非padding处为0.0
        padding = common_attention.attention_bias_to_padding(
          encoder_self_attention_bias)
        nonpadding = 1.0 - padding
        pad_remover = None
        pad_remover = expert_utils.PadRemover(padding)

        # 对每一个encoder隐层
        for layer in range(hparams.num_hidden_layers):
          with tf.variable_scope("layer_%d" % layer):
            with tf.variable_scope("self_attention"):

              # 核心方法，Multihead-Attention
              y = common_attention.multihead_attention(
                  # 先preprocess，进行layer normalization
                  common_layers.layer_preprocess(x, hparams),
                  None,
                  encoder_self_attention_bias,
                  hparams.hidden_size,
                  hparams.hidden_size,
                  hparams.hidden_size,
                  hparams.num_heads,
                  hparams.attention_dropout,
                  attention_type=hparams.self_attention_type,
                  save_weights_to=save_weights_to,
                  max_relative_position=hparams.max_relative_position,
                  max_length=hparams.max_length)
              # 出来[None, None, 512]

              # 对multihead层输出，和multihead输入，进行參差连接(x+y),再dropout
              x = common_layers.layer_postprocess(x, y, hparams)

            # 做FFN，即去掉padding位置向量，逐词做两次全连接，再还原会padding位置
            with tf.variable_scope("ffn"):
              y = self.transformer_ffn_layer(
                  common_layers.layer_preprocess(x, hparams), hparams, pad_remover,
                  conv_padding="SAME", nonpadding_mask=nonpadding,
                  losses=losses)

              # 再做參差连接
              x = common_layers.layer_postprocess(x, y, hparams)
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.

        # 再做normalization
        return common_layers.layer_preprocess(x, hparams)

    def transformer_ffn_layer(self,
                              x,
                              hparams,
                              pad_remover=None,
                              conv_padding="LEFT",
                              nonpadding_mask=None,
                              losses=None,
                              cache=None):

      ffn_layer = hparams.ffn_layer
      relu_dropout_broadcast_dims = (
          common_layers.comma_separated_string_to_integer_list(
              getattr(hparams, "relu_dropout_broadcast_dims", "")))
      if ffn_layer == "conv_hidden_relu":
        # Backwards compatibility
        ffn_layer = "dense_relu_dense"
      if ffn_layer == "dense_relu_dense":
        # In simple convolution mode, use `pad_remover` to speed up processing.
        if pad_remover:
          original_shape = common_layers.shape_list(x)
          # Collapse `x` across examples, and remove padding positions.
          # 变形[batch,seq_len,512] -> [batch*seq_len, 512]
          x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))

          # 将padding位置对应词的向量删除, [batch*seq_len-num_pad, 512]
          x = tf.expand_dims(pad_remover.remove(x), axis=0)

        # 每个词对应的向量做两个全连接, inputs -> filter -> outputs
        # 512 -> 2048 -> 512
        conv_output = common_layers.dense_relu_dense(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            dropout=hparams.relu_dropout,
            dropout_broadcast_dims=relu_dropout_broadcast_dims)

        # 还原成带有padding的数据 [batch, seq_len, 512]
        if pad_remover:
          # Restore `conv_output` to the original shape of `x`, including padding.
          conv_output = tf.reshape(
              pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
        return conv_output
      elif ffn_layer == "conv_relu_conv":
        return common_layers.conv_relu_conv(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            first_kernel_size=hparams.conv_first_kernel,
            second_kernel_size=1,
            padding=conv_padding,
            nonpadding_mask=nonpadding_mask,
            dropout=hparams.relu_dropout,
            cache=cache)
      elif ffn_layer == "parameter_attention":
        return common_attention.parameter_attention(
            x, hparams.parameter_attention_key_channels or hparams.hidden_size,
            hparams.parameter_attention_value_channels or hparams.hidden_size,
            hparams.hidden_size, hparams.filter_size, hparams.num_heads,
            hparams.attention_dropout)
      elif ffn_layer == "conv_hidden_relu_with_sepconv":
        return common_layers.conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            kernel_size=(3, 1),
            second_kernel_size=(31, 1),
            padding="LEFT",
            dropout=hparams.relu_dropout)
      elif ffn_layer == "sru":
        return common_layers.sru(x)
      else:
        assert ffn_layer == "none"
        return x


    def prepare_inputs(self,inputs):

        encoder_input = inputs #[None, None, 512]

        # 获取padding bias
        # 将embedding还原成[None,seq_len],并且padding部分设置为1,其余部分为0
        encoder_padding = common_attention.embedding_to_padding(encoder_input)
        # 将padding部分减去负无穷，再扩展维度,[None, 1, 1, seq_len]
        ignore_padding = common_attention.attention_bias_ignore_padding(
            encoder_padding)
        # 得到padding偏置(用作mask)
        self_attention_bias = ignore_padding

        # 获取其他bias

        # 给输入添加时间(位置)信号
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
        self_attention_bias = tf.cast(self_attention_bias,tf.bfloat16)

        return (encoder_input, self_attention_bias)
