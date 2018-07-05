# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example registrations for T2T."""

from tensor2tensor.utils import registry
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import t2t_model
import tensorflow as tf
from math import pi

# Use register_model for a new T2TModel
# Use register_problem for a new Problem
# Use register_hparams for a new hyperparameter set


@registry.register_model
class TransformerEncoderSimNet(t2t_model.T2TModel):
    """Transformer, encoder only."""
    def body(self, features):
        gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
        hparams = self._hparams
        # enc_input[2 123 1 128] [batch_size max_sentence_len 1 word_enc_size]
        # print([k for k in features])

        in_x = features["input_x"]
        in_y = features["input_y"]
        # in_x = tf.Print(in_x, [tf.shape(in_x), tf.shape(in_y)],"enc_input: ", summarize=100)
        # in_x = tf.Print(in_x, [tf.shape(in_x), tf.shape(in_y)],"enc_input: ", summarize=100)
        # targets = tf.Print(targets, [tf.shape(targets)],"targets shape: ", summarize=100)

        # for k in ['input_y_raw', 'input_x_raw']:
        #     in_raw = features[k]
        #     in_x = tf.Print(in_x, [in_raw],"input_raw: ", summarize=100)

        # in_x = tf.Print(in_x, [targets],"targets: ", summarize=100)
        target_space = features["target_space_id"]

        with tf.variable_scope("single_sentence_encoder") as scope:
            enc_x = sim_encode(in_x, target_space, hparams, features)
            scope.reuse_variables()
            enc_y = sim_encode(in_y, target_space, hparams, features)

        targets = features["targets_raw"]
        enc_out = tf.expand_dims(tf.expand_dims(enc_x, 1),1)
        # if hparams.mode != tf.estimator.ModeKeys.PREDICT:
        enc_sim = tf.reduce_sum(tf.multiply(enc_x, enc_y), axis=1)# tf.tensordot(encoder_output , encoder_output, axes=0)
        enc_sim = 1 - tf.acos(enc_sim) / tf.constant(pi)

        # Finding accuracy:
        ground_truth = tf.reshape(targets, [-1])
        predictions = tf.cast(enc_sim*2,dtype = tf.int32)
        label_weights = tf.cast(tf.reshape(targets, [-1])*(hparams.data_ratio-1)+1,dtype=tf.float32)
        _, acc = tf.metrics.accuracy(ground_truth, predictions, label_weights)
        tf.summary.scalar("Training Accuracy", acc)
        # enc_sim = tf.Print(enc_sim, [gs_t%10,acc_mes],"acc with global step: ", summarize=100)
        # targets = tf.Print(targets, [tf.shape(targets), targets],"loss: ", summarize=100)
        loss = tf.losses.absolute_difference(tf.reshape(targets, [-1]), enc_sim, reduction=tf.losses.Reduction.NONE)
        weighted_loss = tf.losses.compute_weighted_loss(loss, label_weights, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return enc_out, {'training': weighted_loss}


def sim_encode(inputs, target_space, hparams, features):
    # inputs = tf.Print(inputs, [tf.shape(inputs)], "input", summarize=10)
    inputs = common_layers.flatten4d3d(inputs)

    (encoder_input, encoder_self_attention_bias, _) = (
        transformer.transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer.transformer_encoder(
        encoder_input,
        encoder_self_attention_bias,
        hparams,
        nonpadding=transformer.features_to_nonpadding(features, "inputs"))

    positional_mean = tf.nn.l2_normalize(tf.reduce_mean(encoder_output, 1), 1)
    # out_norm = tf.norm(positional_mean)
    # positional_mean = tf.Print(positional_mean , [out_norm], "enc_out: (should be b_size**0.5) ", summarize=10)
    # positional_mean = tf.Print(positional_mean , [tf.shape(positional_mean)], "enc_out: (should be (b_size, h_size)) ", summarize=10)
    return positional_mean


@registry.register_hparams
def transformer_sim_net_tiny():
    hparams = transformer.transformer_base_v2()
    hparams.optimizer_adam_beta2 = 0.997
    hparams.optimizer = "Adam"
    hparams.learning_rate = 0.1
    hparams.learning_rate_warmup_steps = 4000
    hparams.learning_rate_schedule = ("linear_warmup*legacy")
    hparams.num_hidden_layers = 4
    hparams.hidden_size = 256
    hparams.filter_size = 512
    hparams.num_heads = 4
    hparams.batch_size = 4096
    hparams.add_hparam("data_ratio", 4)
    return hparams

@registry.register_hparams
def transformer_sim_net_base():
    hparams = transformer.transformer_base()
    # hparams.optimizer_adam_beta2 = 0.997
    # hparams.learning_rate_constant = 0.1
    # hparams.optimizer = "Adam"
    # hparams.learning_rate = 0.1
    # hparams.learning_rate_warmup_steps = 4000
    # hparams.batch_size = 4096
    hparams.add_hparam("data_ratio", 4)
    return hparams
