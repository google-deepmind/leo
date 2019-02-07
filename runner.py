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
"""A binary building the graph and performing the optimization of LEO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pickle

from absl import flags
import tensorflow as tf

import config
import data
import model
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "/tmp/leo", "Path to restore from and "
                    "save to checkpoints.")
flags.DEFINE_integer(
    "checkpoint_steps", 1000, "The frequency, in number of "
    "steps, of saving the checkpoints.")
flags.DEFINE_boolean("evaluation_mode", False, "Whether to run in an "
                     "evaluation-only mode.")


def _clip_gradients(gradients, gradient_threshold, gradient_norm_threshold):
  """Clips gradients by value and then by norm."""
  if gradient_threshold > 0:
    gradients = [
        tf.clip_by_value(g, -gradient_threshold, gradient_threshold)
        for g in gradients
    ]
  if gradient_norm_threshold > 0:
    gradients = [
        tf.clip_by_norm(g, gradient_norm_threshold) for g in gradients
    ]
  return gradients


def _construct_validation_summaries(metavalid_loss, metavalid_accuracy):
  tf.summary.scalar("metavalid_loss", metavalid_loss)
  tf.summary.scalar("metavalid_valid_accuracy", metavalid_accuracy)
  # The summaries are passed implicitly by TensorFlow.


def _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                  model_grads, model_vars):
  tf.summary.scalar("metatrain_loss", metatrain_loss)
  tf.summary.scalar("metatrain_valid_accuracy", metatrain_accuracy)
  for g, v in zip(model_grads, model_vars):
    histogram_name = v.name.split(":")[0]
    tf.summary.histogram(histogram_name, v)
    histogram_name = "gradient/{}".format(histogram_name)
    tf.summary.histogram(histogram_name, g)


def _construct_examples_batch(batch_size, split, num_classes,
                              num_tr_examples_per_class,
                              num_val_examples_per_class):
  data_provider = data.DataProvider(split, config.get_data_config())
  examples_batch = data_provider.get_batch(batch_size, num_classes,
                                           num_tr_examples_per_class,
                                           num_val_examples_per_class)
  return utils.unpack_data(examples_batch)


def _construct_loss_and_accuracy(inner_model, inputs, is_meta_training):
  """Returns batched loss and accuracy of the model ran on the inputs."""
  call_fn = functools.partial(
      inner_model.__call__, is_meta_training=is_meta_training)
  per_instance_loss, per_instance_accuracy = tf.map_fn(
      call_fn,
      inputs,
      dtype=(tf.float32, tf.float32),
      back_prop=is_meta_training)
  loss = tf.reduce_mean(per_instance_loss)
  accuracy = tf.reduce_mean(per_instance_accuracy)
  return loss, accuracy


def construct_graph(outer_model_config):
  """Constructs the optimization graph."""
  inner_model_config = config.get_inner_model_config()
  tf.logging.info("inner_model_config: {}".format(inner_model_config))
  leo = model.LEO(inner_model_config, use_64bits_dtype=False)

  num_classes = outer_model_config["num_classes"]
  num_tr_examples_per_class = outer_model_config["num_tr_examples_per_class"]
  metatrain_batch = _construct_examples_batch(
      outer_model_config["metatrain_batch_size"], "train", num_classes,
      num_tr_examples_per_class,
      outer_model_config["num_val_examples_per_class"])
  metatrain_loss, metatrain_accuracy = _construct_loss_and_accuracy(
      leo, metatrain_batch, True)

  metatrain_gradients, metatrain_variables = leo.grads_and_vars(metatrain_loss)

  # Avoids NaNs in summaries.
  metatrain_loss = tf.cond(tf.is_nan(metatrain_loss),
                           lambda: tf.zeros_like(metatrain_loss),
                           lambda: metatrain_loss)

  metatrain_gradients = _clip_gradients(
      metatrain_gradients, outer_model_config["gradient_threshold"],
      outer_model_config["gradient_norm_threshold"])

  _construct_training_summaries(metatrain_loss, metatrain_accuracy,
                                metatrain_gradients, metatrain_variables)
  optimizer = tf.train.AdamOptimizer(
      learning_rate=outer_model_config["outer_lr"])
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.apply_gradients(
      zip(metatrain_gradients, metatrain_variables), global_step)

  data_config = config.get_data_config()
  tf.logging.info("data_config: {}".format(data_config))
  total_examples_per_class = data_config["total_examples_per_class"]
  metavalid_batch = _construct_examples_batch(
      outer_model_config["metavalid_batch_size"], "val", num_classes,
      num_tr_examples_per_class,
      total_examples_per_class - num_tr_examples_per_class)
  metavalid_loss, metavalid_accuracy = _construct_loss_and_accuracy(
      leo, metavalid_batch, False)

  metatest_batch = _construct_examples_batch(
      outer_model_config["metatest_batch_size"], "test", num_classes,
      num_tr_examples_per_class,
      total_examples_per_class - num_tr_examples_per_class)
  _, metatest_accuracy = _construct_loss_and_accuracy(
      leo, metatest_batch, False)
  _construct_validation_summaries(metavalid_loss, metavalid_accuracy)

  return (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
          metatest_accuracy)


def run_training_loop(checkpoint_path):
  """Runs the training loop, either saving a checkpoint or evaluating it."""
  outer_model_config = config.get_outer_model_config()
  tf.logging.info("outer_model_config: {}".format(outer_model_config))
  (train_op, global_step, metatrain_accuracy, metavalid_accuracy,
   metatest_accuracy) = construct_graph(outer_model_config)

  num_steps_limit = outer_model_config["num_steps_limit"]
  best_metavalid_accuracy = 0.

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_path,
      save_summaries_steps=FLAGS.checkpoint_steps,
      log_step_count_steps=FLAGS.checkpoint_steps,
      save_checkpoint_steps=FLAGS.checkpoint_steps,
      summary_dir=checkpoint_path) as sess:
    if not FLAGS.evaluation_mode:
      global_step_ev = sess.run(global_step)
      while global_step_ev < num_steps_limit:
        if global_step_ev % FLAGS.checkpoint_steps == 0:
          # Just after saving checkpoint, calculate accuracy 10 times and save
          # the best checkpoint for early stopping.
          metavalid_accuracy_ev = utils.evaluate_and_average(
              sess, metavalid_accuracy, 10)
          tf.logging.info("Step: {} meta-valid accuracy: {}".format(
              global_step_ev, metavalid_accuracy_ev))

          if metavalid_accuracy_ev > best_metavalid_accuracy:
            utils.copy_checkpoint(checkpoint_path, global_step_ev,
                                  metavalid_accuracy_ev)
            best_metavalid_accuracy = metavalid_accuracy_ev

        _, global_step_ev, metatrain_accuracy_ev = sess.run(
            [train_op, global_step, metatrain_accuracy])
        if global_step_ev % (FLAGS.checkpoint_steps // 2) == 0:
          tf.logging.info("Step: {} meta-train accuracy: {}".format(
              global_step_ev, metatrain_accuracy_ev))
    else:
      assert not FLAGS.checkpoint_steps
      num_metatest_estimates = (
          10000 // outer_model_config["metatest_batch_size"])

      test_accuracy = utils.evaluate_and_average(sess, metatest_accuracy,
                                                 num_metatest_estimates)

      tf.logging.info("Metatest accuracy: %f", test_accuracy)
      with tf.gfile.Open(
          os.path.join(checkpoint_path, "test_accuracy"), "wb") as f:
        pickle.dump(test_accuracy, f)


def main(argv):
  del argv  # Unused.
  run_training_loop(FLAGS.checkpoint_path)


if __name__ == "__main__":
  tf.app.run()
