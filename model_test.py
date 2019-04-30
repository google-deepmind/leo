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
"""Tests for ml_leo.model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import parameterized
import mock
import numpy as np
from six.moves import zip
import sonnet as snt
import tensorflow as tf

import data
import model

# Adding float64 and 32 gives an error in TensorFlow.
constant_float64 = lambda x: tf.constant(x, dtype=tf.float64)


def get_test_config():
  """Returns the config used to initialize LEO model."""
  config = {}
  config["inner_unroll_length"] = 3
  config["finetuning_unroll_length"] = 4
  config["inner_lr_init"] = 0.1
  config["finetuning_lr_init"] = 0.2
  config["num_latents"] = 1
  config["dropout_rate"] = 0.3
  config["kl_weight"] = 0.01
  config["encoder_penalty_weight"] = 0.01
  config["l2_penalty_weight"] = 0.01
  config["orthogonality_penalty_weight"] = 0.01

  return config


def mockify_everything(test_function=None,
                       mock_finetuning=True,
                       mock_encdec=True):
  """Mockifies most of the LEO"s model functions to behave as identity."""

  def inner_decorator(f):
    @functools.wraps(f)
    def mockified(*args, **kwargs):
      identity_mapping = lambda unused_self, inp, *args: tf.identity(inp)
      mock_encoder = mock.patch.object(
          model.LEO, "encoder", new=identity_mapping)
      mock_relation_network = mock.patch.object(
          model.LEO, "relation_network", new=identity_mapping)
      mock_decoder = mock.patch.object(
          model.LEO, "decoder", new=identity_mapping)
      mock_average = mock.patch.object(
          model.LEO, "average_codes_per_class", new=identity_mapping)
      mock_loss = mock.patch.object(model.LEO, "loss_fn", new=identity_mapping)

      float64_zero = constant_float64(0.)
      def identity_sample_fn(unused_self, inp, *unused_args, **unused_kwargs):
        return inp, float64_zero

      def mock_sample_with_split(unused_self, inp, *unused_args,
                                 **unused_kwargs):
        out = tf.split(inp, 2, axis=-1)[0]
        return out, float64_zero

      # When not mocking relation net, it will double the latents.
      mock_sample = mock.patch.object(
          model.LEO,
          "possibly_sample",
          new=identity_sample_fn if mock_encdec else mock_sample_with_split)

      def dummy_predict(unused_self, inputs, classifier_weights):
        return inputs * classifier_weights**2

      mock_predict = mock.patch.object(model.LEO, "predict", new=dummy_predict)

      mock_decoder_regularizer = mock.patch.object(
          model.LEO, "_decoder_orthogonality_reg", new=float64_zero)

      all_mocks = [mock_average, mock_loss, mock_predict, mock_sample]
      if mock_encdec:
        all_mocks.extend([
            mock_encoder,
            mock_relation_network,
            mock_decoder,
            mock_decoder_regularizer,
        ])
      if mock_finetuning:
        mock_finetuning_inner = mock.patch.object(
            model.LEO,
            "finetuning_inner_loop",
            new=lambda unused_self, d, l, adapted: (adapted, float64_zero))
        all_mocks.append(mock_finetuning_inner)

      for m in all_mocks:
        m.start()

      f(*args, **kwargs)

      for m in all_mocks:
        m.stop()

    return mockified

  if test_function:
    # Decorator called with no arguments, so the function is passed
    return inner_decorator(test_function)
  return inner_decorator


def _random_problem_instance(num_classes=7,
                             num_examples_per_class=5,
                             embedding_dim=17, use_64bits_dtype=True):
  inputs_dtype = tf.float64 if use_64bits_dtype else tf.float32
  inputs = tf.constant(
      np.random.random((num_classes, num_examples_per_class, embedding_dim)),
      dtype=inputs_dtype)
  outputs_dtype = tf.int64 if use_64bits_dtype else tf.int32
  outputs = tf.constant(
      np.random.randint(
          low=0,
          high=num_classes,
          size=(num_classes, num_examples_per_class, 1)), dtype=outputs_dtype)
  problem = data.ProblemInstance(
      tr_input=inputs,
      val_input=inputs,
      tr_info=inputs,
      tr_output=outputs,
      val_output=outputs,
      val_info=inputs)
  return problem


class LEOTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(LEOTest, self).setUp()
    self._problem = _random_problem_instance(5, 7, 4)
    # This doesn"t call any function, so doesn't need the mocks to be started.
    self._config = get_test_config()
    self._leo = model.LEO(config=self._config)
    self.addCleanup(mock.patch.stopall)

  @mockify_everything
  def test_instantiate_leo(self):
    encoder_output = self._leo.encoder(5, 7)
    with self.session() as sess:
      encoder_output_ev = sess.run(encoder_output)

    self.assertEqual(encoder_output_ev, 5)

  @mockify_everything
  def test_inner_loop_adaptation(self):
    problem_instance = data.ProblemInstance(
        tr_input=constant_float64([[[4.]]]),
        tr_output=tf.constant([[[0]]], dtype=tf.int64),
        tr_info=[],
        val_input=[],
        val_output=[],
        val_info=[],
    )
    # encoder = decoder = id
    # predict returns classifier_weights**2 * inputs = latents**2 * inputs
    # loss = id = inputs*latents
    # dl/dlatent = 2 * latent * inputs
    # 4 -> 4 - 0.1 * 2 * 4 * 4 = 0.8
    # 0.8 -> 0.8 - 0.1 * 2 * 0.8 * 4 = 0.16
    # 0.16 -> 0.16 - 0.1 * 2 * 0.16 * 4 = 0.032

    # is_meta_training=False disables kl and encoder penalties
    adapted_parameters, _ = self._leo(problem_instance, is_meta_training=False)

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(adapted_parameters), 0.032)

  @mockify_everything
  def test_map_input(self):
    problem = [
        constant_float64([[[5.]]]),  # tr_input
        tf.constant([[[0]]], dtype=tf.int64),  # tr_output
        constant_float64([[[0]]]),  # tr_info
        constant_float64([[[0.]]]),  # val_input
        tf.constant([[[0]]], dtype=tf.int64),  # val_output
        constant_float64([[[0]]]),  # val_info
    ]
    another_problem = [
        constant_float64([[[4.]]]),
        tf.constant([[[0]]], dtype=tf.int64),
        constant_float64([[[0]]]),
        constant_float64([[[0.]]]),
        tf.constant([[[0]]], dtype=tf.int64),
        constant_float64([[[0]]]),
    ]
    # first dimension (list): diffent input kind (tr_input, val_output, etc.)
    # second dim: different problems; this has to be a tensor dim for map_fn
    # to split over it.
    # next three: (1, 1, 1)

    # map_fn cannot receive structured inputs (namedtuples).
    ins = [
        tf.stack([in1, in2])
        for in1, in2 in zip(problem, another_problem)
    ]

    two_adapted_params, _ = tf.map_fn(
        self._leo.__call__, ins, dtype=(tf.float64, tf.float64))

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      output1, output2 = sess.run(two_adapted_params)
      self.assertGreater(abs(output1 - output2), 1e-3)

  @mockify_everything
  def test_setting_is_meta_training(self):
    self._leo(self._problem, is_meta_training=True)
    self.assertTrue(self._leo.is_meta_training)
    self._leo(self._problem, is_meta_training=False)
    self.assertFalse(self._leo.is_meta_training)

  @mockify_everything(mock_finetuning=False)
  def test_finetuning_improves_loss(self):
    # Create graph
    self._leo(self._problem)

    latents, _ = self._leo.forward_encoder(self._problem)
    leo_loss, adapted_classifier_weights, _ = self._leo.leo_inner_loop(
        self._problem, latents)
    leo_loss = tf.reduce_mean(leo_loss)
    finetuning_loss, _ = self._leo.finetuning_inner_loop(
        self._problem, leo_loss, adapted_classifier_weights)
    finetuning_loss = tf.reduce_mean(finetuning_loss)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      leo_loss_ev, finetuning_loss_ev = sess.run([leo_loss, finetuning_loss])
      self.assertGreater(leo_loss_ev - 1e-3, finetuning_loss_ev)

  @mockify_everything
  def test_gradients_dont_flow_through_input(self):
    # Create graph
    self._leo(self._problem)
    latents, _ = self._leo.forward_encoder(self._problem)
    grads = tf.gradients(self._problem.tr_input, latents)
    self.assertIsNone(grads[0])

  @mockify_everything
  def test_inferring_embedding_dim(self):
    self._leo(self._problem)
    self.assertEqual(self._leo.embedding_dim, 4)

  @mockify_everything(mock_encdec=False, mock_finetuning=False)
  def test_variable_creation(self):
    self._leo(self._problem)
    encoder_variables = snt.get_variables_in_scope("leo/encoder")
    self.assertNotEmpty(encoder_variables)
    relation_network_variables = snt.get_variables_in_scope(
        "leo/relation_network")
    self.assertNotEmpty(relation_network_variables)
    decoder_variables = snt.get_variables_in_scope("leo/decoder")
    self.assertNotEmpty(decoder_variables)
    inner_lr = snt.get_variables_in_scope("leo/leo_inner")
    self.assertNotEmpty(inner_lr)
    finetuning_lr = snt.get_variables_in_scope("leo/finetuning")
    self.assertNotEmpty(finetuning_lr)
    self.assertSameElements(
        encoder_variables + relation_network_variables + decoder_variables +
        inner_lr + finetuning_lr, self._leo.trainable_variables)

  def test_graph_construction(self):
    self._leo(self._problem)

  def test_possibly_sample(self):
    # Embedding dimension has to be divisible by 2 here.
    self._leo(self._problem, is_meta_training=True)
    train_samples, train_kl = self._leo.possibly_sample(self._problem.tr_input)

    self._leo(self._problem, is_meta_training=False)
    test_samples, test_kl = self._leo.possibly_sample(self._problem.tr_input)

    with self.session() as sess:
      train_samples_ev1, test_samples_ev1 = sess.run(
          [train_samples, test_samples])
      train_samples_ev2, test_samples_ev2 = sess.run(
          [train_samples, test_samples])

      self.assertAllClose(test_samples_ev1, test_samples_ev2)
      self.assertGreater(abs(np.sum(train_samples_ev1 - train_samples_ev2)), 1.)

      train_kl_ev, test_kl_ev = sess.run([train_kl, test_kl])
      self.assertNotEqual(train_kl_ev, 0.)
      self.assertEqual(test_kl_ev, 0.)

  def test_different_shapes(self):
    problem_instance2 = _random_problem_instance(5, 6, 13)

    self._leo(self._problem)
    with self.assertRaises(AssertionError):
      self._leo(problem_instance2)

  def test_encoder_penalty(self):
    self._leo(self._problem)  # Sets is_meta_training
    latents, _ = self._leo.forward_encoder(self._problem)
    _, _, train_encoder_penalty = self._leo.leo_inner_loop(
        self._problem, latents)

    self._leo(self._problem, is_meta_training=False)
    _, _, test_encoder_penalty = self._leo.leo_inner_loop(
        self._problem, latents)

    with self.session() as sess:
      sess.run(tf.initializers.global_variables())
      train_encoder_penalty_ev, test_encoder_penalty_ev = sess.run(
          [train_encoder_penalty, test_encoder_penalty])
      self.assertGreater(train_encoder_penalty_ev, 1e-3)
      self.assertLess(test_encoder_penalty_ev, 1e-7)

  def test_construct_float32_leo_graph(self):
    leo = model.LEO(use_64bits_dtype=False, config=self._config)
    problem_instance_32_bits = _random_problem_instance(use_64bits_dtype=False)
    leo(problem_instance_32_bits)


if __name__ == "__main__":
  tf.test.main()
