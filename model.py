# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Code defining LEO inner loop.

See "Meta-Learning with Latent Embedding Optimization" by Rusu et al.
(https://arxiv.org/pdf/1807.05960.pdf).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

import data as data_module


def get_orthogonality_regularizer(orthogonality_penalty_weight):
  """Returns the orthogonality regularizer."""
  def orthogonality(weight):
    """Calculates the layer-wise penalty encouraging orthogonality."""
    with tf.name_scope(None, "orthogonality", [weight]) as name:
      w2 = tf.matmul(weight, weight, transpose_b=True)
      wn = tf.norm(weight, ord=2, axis=1, keepdims=True) + 1e-32
      correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
      matrix_size = correlation_matrix.get_shape().as_list()[0]
      base_dtype = weight.dtype.base_dtype
      identity = tf.eye(matrix_size, dtype=base_dtype)
      weight_corr = tf.reduce_mean(
          tf.squared_difference(correlation_matrix, identity))
      return tf.multiply(
          tf.cast(orthogonality_penalty_weight, base_dtype),
          weight_corr,
          name=name)

  return orthogonality


class LEO(snt.AbstractModule):
  """Sonnet module implementing the inner loop of LEO."""

  def __init__(self, config=None, use_64bits_dtype=True, name="leo"):
    super(LEO, self).__init__(name=name)

    self._float_dtype = tf.float64 if use_64bits_dtype else tf.float32
    self._int_dtype = tf.int64 if use_64bits_dtype else tf.int32

    self._inner_unroll_length = config["inner_unroll_length"]
    self._finetuning_unroll_length = config["finetuning_unroll_length"]
    self._inner_lr_init = config["inner_lr_init"]
    self._finetuning_lr_init = config["finetuning_lr_init"]
    self._num_latents = config["num_latents"]
    self._dropout_rate = config["dropout_rate"]

    self._kl_weight = config["kl_weight"]  # beta
    self._encoder_penalty_weight = config["encoder_penalty_weight"]  # gamma
    self._l2_penalty_weight = config["l2_penalty_weight"]  # lambda_1
    # lambda_2
    self._orthogonality_penalty_weight = config["orthogonality_penalty_weight"]

    assert self._inner_unroll_length > 0, ("Positive unroll length is necessary"
                                           " to create the graph")

  def _build(self, data, is_meta_training=True):
    """Connects the LEO module to the graph, creating the variables.

    Args:
      data: A data_module.ProblemInstance constaining Tensors with the
          following shapes:
          - tr_input: (N, K, dim)
          - tr_output: (N, K, 1)
          - tr_info: (N, K)
          - val_input: (N, K_valid, dim)
          - val_output: (N, K_valid, 1)
          - val_info: (N, K_valid)
            where N is the number of classes (as in N-way) and K and the and
            K_valid are numbers of training and validation examples within a
            problem instance correspondingly (as in K-shot), and dim is the
            dimensionality of the embedding.
      is_meta_training: A boolean describing whether we run in the training
        mode.

    Returns:
      Tensor with the inner validation loss of LEO (include both adaptation in
      the latent space and finetuning).
    """
    if isinstance(data, list):
      data = data_module.ProblemInstance(*data)
    self.is_meta_training = is_meta_training
    self.save_problem_instance_stats(data.tr_input)

    latents, kl = self.forward_encoder(data)
    tr_loss, adapted_classifier_weights, encoder_penalty = self.leo_inner_loop(
        data, latents)

    val_loss, val_accuracy = self.finetuning_inner_loop(
        data, tr_loss, adapted_classifier_weights)

    val_loss += self._kl_weight * kl
    val_loss += self._encoder_penalty_weight * encoder_penalty
    # The l2 regularization is is already added to the graph when constructing
    # the snt.Linear modules. We pass the orthogonality regularizer separately,
    # because it is not used in self.grads_and_vars.
    regularization_penalty = (
        self._l2_regularization + self._decoder_orthogonality_reg)

    batch_val_loss = tf.reduce_mean(val_loss)
    batch_val_accuracy = tf.reduce_mean(val_accuracy)

    return batch_val_loss + regularization_penalty, batch_val_accuracy

  @snt.reuse_variables
  def leo_inner_loop(self, data, latents):
    with tf.variable_scope("leo_inner"):
      inner_lr = tf.get_variable(
          "lr", [1, 1, self._num_latents],
          dtype=self._float_dtype,
          initializer=tf.constant_initializer(self._inner_lr_init))
    starting_latents = latents
    loss, _ = self.forward_decoder(data, latents)
    for _ in xrange(self._inner_unroll_length):
      loss_grad = tf.gradients(loss, latents)  # dLtrain/dz
      latents -= inner_lr * loss_grad[0]
      loss, classifier_weights = self.forward_decoder(data, latents)

    if self.is_meta_training:
      encoder_penalty = tf.losses.mean_squared_error(
          labels=tf.stop_gradient(latents), predictions=starting_latents)
      encoder_penalty = tf.cast(encoder_penalty, self._float_dtype)
    else:
      encoder_penalty = tf.constant(0., self._float_dtype)

    return loss, classifier_weights, encoder_penalty

  @snt.reuse_variables
  def finetuning_inner_loop(self, data, leo_loss, classifier_weights):
    tr_loss = leo_loss
    with tf.variable_scope("finetuning"):
      finetuning_lr = tf.get_variable(
          "lr", [1, 1, self.embedding_dim],
          dtype=self._float_dtype,
          initializer=tf.constant_initializer(self._finetuning_lr_init))
    for _ in xrange(self._finetuning_unroll_length):
      loss_grad = tf.gradients(tr_loss, classifier_weights)
      classifier_weights -= finetuning_lr * loss_grad[0]
      tr_loss, _ = self.calculate_inner_loss(data.tr_input, data.tr_output,
                                             classifier_weights)

    val_loss, val_accuracy = self.calculate_inner_loss(
        data.val_input, data.val_output, classifier_weights)
    return val_loss, val_accuracy

  @snt.reuse_variables
  def forward_encoder(self, data):
    encoder_outputs = self.encoder(data.tr_input)
    relation_network_outputs = self.relation_network(encoder_outputs)
    latent_dist_params = self.average_codes_per_class(relation_network_outputs)
    latents, kl = self.possibly_sample(latent_dist_params)
    return latents, kl

  @snt.reuse_variables
  def forward_decoder(self, data, latents):
    weights_dist_params = self.decoder(latents)
    # Default to glorot_initialization and not stddev=1.
    fan_in = self.embedding_dim.value
    fan_out = self.num_classes.value
    stddev_offset = np.sqrt(2. / (fan_out + fan_in))
    classifier_weights, _ = self.possibly_sample(weights_dist_params,
                                                 stddev_offset=stddev_offset)
    tr_loss, _ = self.calculate_inner_loss(data.tr_input, data.tr_output,
                                           classifier_weights)
    return tr_loss, classifier_weights

  @snt.reuse_variables
  def encoder(self, inputs):
    with tf.variable_scope("encoder"):
      after_dropout = tf.nn.dropout(inputs, rate=self.dropout_rate)
      regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
      initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
      encoder_module = snt.Linear(
          self._num_latents,
          use_bias=False,
          regularizers={"w": regularizer},
          initializers={"w": initializer},
      )
      outputs = snt.BatchApply(encoder_module)(after_dropout)
      return outputs

  @snt.reuse_variables
  def relation_network(self, inputs):
    with tf.variable_scope("relation_network"):
      regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
      initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
      relation_network_module = snt.nets.MLP(
          [2 * self._num_latents] * 3,
          use_bias=False,
          regularizers={"w": regularizer},
          initializers={"w": initializer},
      )
      total_num_examples = self.num_examples_per_class*self.num_classes
      inputs = tf.reshape(inputs, [total_num_examples, self._num_latents])

      left = tf.tile(tf.expand_dims(inputs, 1), [1, total_num_examples, 1])
      right = tf.tile(tf.expand_dims(inputs, 0), [total_num_examples, 1, 1])
      concat_codes = tf.concat([left, right], axis=-1)
      outputs = snt.BatchApply(relation_network_module)(concat_codes)
      outputs = tf.reduce_mean(outputs, axis=1)
      # 2 * latents, because we are returning means and variances of a Gaussian
      outputs = tf.reshape(outputs, [self.num_classes,
                                     self.num_examples_per_class,
                                     2 * self._num_latents])

      return outputs

  @snt.reuse_variables
  def decoder(self, inputs):
    with tf.variable_scope("decoder"):
      l2_regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
      orthogonality_reg = get_orthogonality_regularizer(
          self._orthogonality_penalty_weight)
      initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
      # 2 * embedding_dim, because we are returning means and variances
      decoder_module = snt.Linear(
          2 * self.embedding_dim,
          use_bias=False,
          regularizers={"w": l2_regularizer},
          initializers={"w": initializer},
      )
      outputs = snt.BatchApply(decoder_module)(inputs)
      self._orthogonality_reg = orthogonality_reg(decoder_module.w)
      return outputs

  def average_codes_per_class(self, codes):
    codes = tf.reduce_mean(codes, axis=1, keep_dims=True)  # K dimension
    # Keep the shape (N, K, *)
    codes = tf.tile(codes, [1, self.num_examples_per_class, 1])
    return codes

  def possibly_sample(self, distribution_params, stddev_offset=0.):
    means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1)
    stddev = tf.exp(unnormalized_stddev)
    stddev -= (1. - stddev_offset)
    stddev = tf.maximum(stddev, 1e-10)
    distribution = tfp.distributions.Normal(loc=means, scale=stddev)
    if not self.is_meta_training:
      return means, tf.constant(0., dtype=self._float_dtype)

    samples = distribution.sample()
    kl_divergence = self.kl_divergence(samples, distribution)
    return samples, kl_divergence

  def kl_divergence(self, samples, normal_distribution):
    random_prior = tfp.distributions.Normal(
        loc=tf.zeros_like(samples), scale=tf.ones_like(samples))
    kl = tf.reduce_mean(
        normal_distribution.log_prob(samples) - random_prior.log_prob(samples))
    return kl

  def predict(self, inputs, weights):
    after_dropout = tf.nn.dropout(inputs, rate=self.dropout_rate)
    # This is 3-dimensional equivalent of a matrix product, where we sum over
    # the last (embedding_dim) dimension. We get [N, K, N, K] tensor as output.
    per_image_predictions = tf.einsum("ijk,lmk->ijlm", after_dropout, weights)

    # Predictions have shape [N, K, N]: for each image ([N, K] of them), what
    # is the probability of a given class (N)?
    predictions = tf.reduce_mean(per_image_predictions, axis=-1)
    return predictions

  def calculate_inner_loss(self, inputs, true_outputs, classifier_weights):
    model_outputs = self.predict(inputs, classifier_weights)
    model_predictions = tf.argmax(
        model_outputs, -1, output_type=self._int_dtype)
    accuracy = tf.contrib.metrics.accuracy(model_predictions,
                                           tf.squeeze(true_outputs, axis=-1))

    return self.loss_fn(model_outputs, true_outputs), accuracy

  def save_problem_instance_stats(self, instance):
    num_classes, num_examples_per_class, embedding_dim = instance.get_shape()
    if hasattr(self, "num_classes"):
      assert self.num_classes == num_classes, (
          "Given different number of classes (N in N-way) in consecutive runs.")
    if hasattr(self, "num_examples_per_class"):
      assert self.num_examples_per_class == num_examples_per_class, (
          "Given different number of examples (K in K-shot) in consecutive"
          "runs.")
    if hasattr(self, "embedding_dim"):
      assert self.embedding_dim == embedding_dim, (
          "Given different embedding dimension in consecutive runs.")

    self.num_classes = num_classes
    self.num_examples_per_class = num_examples_per_class
    self.embedding_dim = embedding_dim

  @property
  def dropout_rate(self):
    return self._dropout_rate if self.is_meta_training else 0.0

  def loss_fn(self, model_outputs, original_classes):
    original_classes = tf.squeeze(original_classes, axis=-1)
    # Tensorflow doesn't handle second order gradients of a sparse_softmax yet.
    one_hot_outputs = tf.one_hot(original_classes, depth=self.num_classes)
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_outputs, logits=model_outputs)

  def grads_and_vars(self, metatrain_loss):
    """Computes gradients of metatrain_loss, avoiding NaN.

    Uses a fixed penalty of 1e-4 to enforce only the l2 regularization (and not
    minimize the loss) when metatrain_loss or any of its gradients with respect
    to trainable_vars are NaN. In practice, this approach pulls the variables
    back into a feasible region of the space when the loss or its gradients are
    not defined.

    Args:
      metatrain_loss: A tensor with the LEO meta-training loss.

    Returns:
      A tuple with:
        metatrain_gradients: A list of gradient tensors.
        metatrain_variables: A list of variables for this LEO model.
    """
    metatrain_variables = self.trainable_variables
    metatrain_gradients = tf.gradients(metatrain_loss, metatrain_variables)

    nan_loss_or_grad = tf.logical_or(
        tf.is_nan(metatrain_loss),
        tf.reduce_any([tf.reduce_any(tf.is_nan(g))
                       for g in metatrain_gradients]))

    regularization_penalty = (
        1e-4 / self._l2_penalty_weight * self._l2_regularization)
    zero_or_regularization_gradients = [
        g if g is not None else tf.zeros_like(v)
        for v, g in zip(tf.gradients(regularization_penalty,
                                     metatrain_variables), metatrain_variables)]

    metatrain_gradients = tf.cond(nan_loss_or_grad,
                                  lambda: zero_or_regularization_gradients,
                                  lambda: metatrain_gradients, strict=True)

    return metatrain_gradients, metatrain_variables

  @property
  def _l2_regularization(self):
    return tf.cast(
        tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
        dtype=self._float_dtype)

  @property
  def _decoder_orthogonality_reg(self):
    return self._orthogonality_reg
