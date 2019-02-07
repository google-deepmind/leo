# coding=utf8
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
"""A module containing just the configs for the different LEO parts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Path to the dataset.")
flags.DEFINE_string(
    "dataset_name", "miniImageNet", "Name of the dataset to "
    "train on, which will be mapped to data.MetaDataset.")
flags.DEFINE_string(
    "embedding_crop", "center", "Type of the cropping, which "
    "will be mapped to data.EmbeddingCrop.")
flags.DEFINE_boolean("train_on_val", False, "Whether to train on the "
                     "validation data.")

flags.DEFINE_integer(
    "inner_unroll_length", 5, "Number of unroll steps in the "
    "inner loop of leo (number of adaptation steps in the "
    "latent space).")
flags.DEFINE_integer(
    "finetuning_unroll_length", 5, "Number of unroll steps "
    "in the loop performing finetuning (number of adaptation "
    "steps in the parameter space).")
flags.DEFINE_integer("num_latents", 64, "The dimensionality of the latent "
                     "space.")
flags.DEFINE_float(
    "inner_lr_init", 1.0, "The initialization value for the "
    "learning rate of the inner loop of leo.")
flags.DEFINE_float(
    "finetuning_lr_init", 0.001, "The initialization value for "
    "learning rate of the finetuning loop.")
flags.DEFINE_float("dropout_rate", 0.5, "Rate of dropout: probability of "
                   "dropping a given unit.")
flags.DEFINE_float(
    "kl_weight", 1e-3, "The weight measuring importance of the "
    "KL in the final loss. β in the paper.")
flags.DEFINE_float(
    "encoder_penalty_weight", 1e-9, "The weight measuring "
    "importance of the encoder penalty in the final loss. γ in "
    "the paper.")
flags.DEFINE_float("l2_penalty_weight", 1e-8, "The weight measuring the "
                   "importance of the l2 regularization in the final loss. λ₁ "
                   "in the paper.")
flags.DEFINE_float("orthogonality_penalty_weight", 1e-3, "The weight measuring "
                   "the importance of the decoder orthogonality regularization "
                   "in the final loss. λ₂ in the paper.")

flags.DEFINE_integer(
    "num_classes", 5, "Number of classes, N in N-way classification.")
flags.DEFINE_integer(
    "num_tr_examples_per_class", 1, "Number of training samples per class, "
    "K in K-shot classification.")
flags.DEFINE_integer(
    "num_val_examples_per_class", 15, "Number of validation samples per class "
    "in a task instance.")
flags.DEFINE_integer("metatrain_batch_size", 12, "Number of problem instances "
                     "in a batch.")
flags.DEFINE_integer("metavalid_batch_size", 200, "Number of meta-validation "
                     "problem instances.")
flags.DEFINE_integer("metatest_batch_size", 200, "Number of meta-testing "
                     "problem instances.")
flags.DEFINE_integer("num_steps_limit", int(1e5), "Number of steps to train "
                     "for.")
flags.DEFINE_float("outer_lr", 1e-4, "Outer (metatraining) loop learning "
                   "rate.")
flags.DEFINE_float(
    "gradient_threshold", 0.1, "The cutoff for the gradient "
    "clipping. Gradients will be clipped to "
    "[-gradient_threshold, gradient_threshold]")
flags.DEFINE_float(
    "gradient_norm_threshold", 0.1, "The cutoff for clipping of "
    "the gradient norm. Gradient norm clipping will be applied "
    "after pointwise clipping (described above).")


def get_data_config():
  config = {}
  config["data_path"] = FLAGS.data_path
  config["dataset_name"] = FLAGS.dataset_name
  config["embedding_crop"] = FLAGS.embedding_crop
  config["train_on_val"] = FLAGS.train_on_val
  config["total_examples_per_class"] = 600
  return config


def get_inner_model_config():
  """Returns the config used to initialize LEO model."""
  config = {}
  config["inner_unroll_length"] = FLAGS.inner_unroll_length
  config["finetuning_unroll_length"] = FLAGS.finetuning_unroll_length
  config["num_latents"] = FLAGS.num_latents
  config["inner_lr_init"] = FLAGS.inner_lr_init
  config["finetuning_lr_init"] = FLAGS.finetuning_lr_init
  config["dropout_rate"] = FLAGS.dropout_rate
  config["kl_weight"] = FLAGS.kl_weight
  config["encoder_penalty_weight"] = FLAGS.encoder_penalty_weight
  config["l2_penalty_weight"] = FLAGS.l2_penalty_weight
  config["orthogonality_penalty_weight"] = FLAGS.orthogonality_penalty_weight

  return config


def get_outer_model_config():
  """Returns the outer config file for N-way K-shot classification tasks."""
  config = {}
  config["num_classes"] = FLAGS.num_classes
  config["num_tr_examples_per_class"] = FLAGS.num_tr_examples_per_class
  config["num_val_examples_per_class"] = FLAGS.num_val_examples_per_class
  config["metatrain_batch_size"] = FLAGS.metatrain_batch_size
  config["metavalid_batch_size"] = FLAGS.metavalid_batch_size
  config["metatest_batch_size"] = FLAGS.metatest_batch_size
  config["num_steps_limit"] = FLAGS.num_steps_limit
  config["outer_lr"] = FLAGS.outer_lr
  config["gradient_threshold"] = FLAGS.gradient_threshold
  config["gradient_norm_threshold"] = FLAGS.gradient_norm_threshold
  return config
