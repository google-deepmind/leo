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
"""Short utility functions for LEO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import tensorflow as tf

import config
import data


def unpack_data(problem_instance):
  """Map data.ProblemInstance to a list of Tensors, to process with map_fn."""
  if isinstance(problem_instance, data.ProblemInstance):
    return list(problem_instance)
  return problem_instance


def copy_checkpoint(checkpoint_path, global_step, accuracy):
  """Copies the checkpoint to a separate directory."""
  tmp_checkpoint_path = os.path.join(checkpoint_path, "tmp_best_checkpoint")
  best_checkpoint_path = os.path.join(checkpoint_path, "best_checkpoint")
  if _is_previous_accuracy_better(best_checkpoint_path, accuracy):
    tf.logging.info("Not copying the checkpoint: there is a better one from "
                    "before a preemption.")
    return

  checkpoint_regex = os.path.join(checkpoint_path,
                                  "model.ckpt-{}.*".format(global_step))
  checkpoint_files = tf.gfile.Glob(checkpoint_regex)
  graph_file = os.path.join(checkpoint_path, "graph.pbtxt")
  checkpoint_files.append(graph_file)

  _save_files_in_tmp_directory(tmp_checkpoint_path, checkpoint_files, accuracy)

  new_checkpoint_index_file = os.path.join(tmp_checkpoint_path, "checkpoint")
  with tf.gfile.Open(new_checkpoint_index_file, "w") as f:
    f.write("model_checkpoint_path: \"{}/model.ckpt-{}\"\n".format(
        best_checkpoint_path, global_step))

  # We first copy the better checkpoint to a temporary directory, and only
  # when it's created move it to avoid inconsistent state when job is preempted
  # when copying the checkpoint.
  if tf.gfile.Exists(best_checkpoint_path):
    tf.gfile.DeleteRecursively(best_checkpoint_path)
  tf.gfile.Rename(tmp_checkpoint_path, best_checkpoint_path)
  tf.logging.info("Copied new best checkpoint with accuracy %.5f", accuracy)


def _save_files_in_tmp_directory(tmp_checkpoint_path, checkpoint_files,
                                 accuracy):
  """Saves the checkpoint files and accuracy in a temporary directory."""

  if tf.gfile.Exists(tmp_checkpoint_path):
    tf.logging.info("The temporary directory exists, because job was preempted "
                    "before it managed to move it. We're removing it.")
    tf.gfile.DeleteRecursively(tmp_checkpoint_path)
  tf.gfile.MkDir(tmp_checkpoint_path)

  def dump_in_best_checkpoint_path(obj, filename):
    full_path = os.path.join(tmp_checkpoint_path, filename)
    with tf.gfile.Open(full_path, "wb") as f:
      pickle.dump(obj, f)

  for file_ in checkpoint_files:
    just_filename = file_.split("/")[-1]
    tf.gfile.Copy(
        file_,
        os.path.join(tmp_checkpoint_path, just_filename),
        overwrite=False)
  dump_in_best_checkpoint_path(config.get_inner_model_config(), "inner_config")
  dump_in_best_checkpoint_path(config.get_outer_model_config(), "outer_config")
  dump_in_best_checkpoint_path(accuracy, "accuracy")


def _is_previous_accuracy_better(best_checkpoint_path, accuracy):
  if not tf.gfile.Exists(best_checkpoint_path):
    return False

  previous_accuracy_file = os.path.join(best_checkpoint_path, "accuracy")
  with tf.gfile.Open(previous_accuracy_file, "rb") as f:
    previous_accuracy = pickle.load(f)

  return previous_accuracy > accuracy


def evaluate_and_average(session, tensor, num_estimates):
  tensor_value_estimates = [session.run(tensor) for _ in xrange(num_estimates)]
  average_tensor_value = sum(tensor_value_estimates) / num_estimates
  return average_tensor_value
