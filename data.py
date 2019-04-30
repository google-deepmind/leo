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
"""Creates problem instances for LEO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
import random

import enum
import numpy as np
import six
import tensorflow as tf


NDIM = 640

ProblemInstance = collections.namedtuple(
    "ProblemInstance",
    ["tr_input", "tr_output", "tr_info", "val_input", "val_output", "val_info"])


class StrEnum(enum.Enum):
  """An Enum represented by a string."""

  def __str__(self):
    return self.value

  def __repr__(self):
    return self.__str__()


class MetaDataset(StrEnum):
  """Datasets supported by the DataProvider class."""
  MINI = "miniImageNet"
  TIERED = "tieredImageNet"


class EmbeddingCrop(StrEnum):
  """Embedding types supported by the DataProvider class."""
  CENTER = "center"
  MULTIVIEW = "multiview"


class MetaSplit(StrEnum):
  """Meta-datasets split supported by the DataProvider class."""
  TRAIN = "train"
  VALID = "val"
  TEST = "test"


class DataProvider(object):
  """Creates problem instances from a specific split and dataset."""

  def __init__(self, dataset_split, config, verbose=False):
    self._dataset_split = MetaSplit(dataset_split)
    self._config = config
    self._verbose = verbose
    self._check_config()

    self._index_data(self._load_data())

  def _check_config(self):
    """Checks configuration arguments of constructor."""
    self._config["dataset_name"] = MetaDataset(self._config["dataset_name"])
    self._config["embedding_crop"] = EmbeddingCrop(
        self._config["embedding_crop"])
    if self._config["dataset_name"] == MetaDataset.TIERED:
      error_message = "embedding_crop: {} not supported for {}".format(
          self._config["embedding_crop"], self._config["dataset_name"])
      assert self._config[
          "embedding_crop"] == EmbeddingCrop.CENTER, error_message

  def _load_data(self):
    """Loads data into memory and caches ."""
    raw_data = self._load(
        tf.gfile.Open(self._get_full_pickle_path(self._dataset_split), "rb"))
    if self._dataset_split == MetaSplit.TRAIN and self._config["train_on_val"]:
      valid_data = self._load(
          tf.gfile.Open(self._get_full_pickle_path(MetaSplit.VALID), "rb"))
      for key in valid_data:
        if self._verbose:
          tf.logging.info(str([key, raw_data[key].shape]))
        raw_data[key] = np.concatenate([raw_data[key],
                                        valid_data[key]], axis=0)
        if self._verbose:
          tf.logging.info(str([key, raw_data[key].shape]))

    if self._verbose:
      tf.logging.info(
          str([(k, np.shape(v)) for k, v in six.iteritems(raw_data)]))

    return raw_data

  def _load(self, opened_file):
    if six.PY2:
      result = pickle.load(opened_file)
    else:
      result = pickle.load(opened_file, encoding="latin1")  # pylint: disable=unexpected-keyword-arg
    return result

  def _index_data(self, raw_data):
    """Builds an index of images embeddings by class."""
    self._all_class_images = collections.OrderedDict()
    self._image_embedding = collections.OrderedDict()
    for i, k in enumerate(raw_data["keys"]):
      _, class_label, image_file = k.split("-")
      image_file_class_label = image_file.split("_")[0]
      assert class_label == image_file_class_label
      self._image_embedding[image_file] = raw_data["embeddings"][i]
      if class_label not in self._all_class_images:
        self._all_class_images[class_label] = []
      self._all_class_images[class_label].append(image_file)

    self._check_data_index(raw_data)

    self._all_class_images = collections.OrderedDict([
        (k, np.array(v)) for k, v in six.iteritems(self._all_class_images)
    ])
    if self._verbose:
      tf.logging.info(str([len(raw_data), len(self._all_class_images),
                           len(self._image_embedding)]))

  def _check_data_index(self, raw_data):
    """Performs checks of the data index and image counts per class."""
    n = raw_data["keys"].shape[0]
    error_message = "{} != {}".format(len(self._image_embedding), n)
    assert len(self._image_embedding) == n, error_message
    error_message = "{} != {}".format(raw_data["embeddings"].shape[0], n)
    assert raw_data["embeddings"].shape[0] == n, error_message

    all_class_folders = list(self._all_class_images.keys())
    error_message = "no duplicate class names"
    assert len(set(all_class_folders)) == len(all_class_folders), error_message
    image_counts = set([len(class_images)
                        for class_images in self._all_class_images.values()])
    error_message = ("len(image_counts) should have at least one element but "
                     "is: {}").format(image_counts)
    assert len(image_counts) >= 1, error_message
    assert min(image_counts) > 0

  def _get_full_pickle_path(self, split_name):
    full_pickle_path = os.path.join(
        self._config["data_path"],
        str(self._config["dataset_name"]),
        str(self._config["embedding_crop"]),
        "{}_embeddings.pkl".format(split_name))
    if self._verbose:
      tf.logging.info("get_one_emb_instance: folder_path: {}".format(
          full_pickle_path))
    return full_pickle_path

  def get_instance(self, num_classes, tr_size, val_size):
    """Samples a random N-way K-shot classification problem instance.

    Args:
      num_classes: N in N-way classification.
      tr_size: K in K-shot; number of training examples per class.
      val_size: number of validation examples per class.

    Returns:
      A tuple with 6 Tensors with the following shapes:
      - tr_input: (num_classes, tr_size, NDIM): training image embeddings.
      - tr_output: (num_classes, tr_size, 1): training image labels.
      - tr_info: (num_classes, tr_size): training image file names.
      - val_input: (num_classes, val_size, NDIM): validation image embeddings.
      - val_output: (num_classes, val_size, 1): validation image labels.
      - val_input: (num_classes, val_size): validation image file names.
    """

    def _build_one_instance_py():
      """Builds a random problem instance using data from specified classes."""
      class_list = list(self._all_class_images.keys())
      sample_count = (tr_size + val_size)
      shuffled_folders = class_list[:]
      random.shuffle(shuffled_folders)
      shuffled_folders = shuffled_folders[:num_classes]
      error_message = "len(shuffled_folders) {} is not num_classes: {}".format(
          len(shuffled_folders), num_classes)
      assert len(shuffled_folders) == num_classes, error_message
      image_paths = []
      class_ids = []
      embeddings = self._image_embedding
      for class_id, class_name in enumerate(shuffled_folders):
        all_images = self._all_class_images[class_name]
        all_images = np.random.choice(all_images, sample_count, replace=False)
        error_message = "{} == {} failed".format(len(all_images), sample_count)
        assert len(all_images) == sample_count, error_message
        image_paths.append(all_images)
        class_ids.append([[class_id]]*sample_count)

      label_array = np.array(class_ids, dtype=np.int32)
      if self._verbose:
        tf.logging.info(label_array.shape)
      if self._verbose:
        tf.logging.info(label_array.shape)
      path_array = np.array(image_paths)
      if self._verbose:
        tf.logging.info(path_array.shape)
      if self._verbose:
        tf.logging.info(path_array.shape)
      embedding_array = np.array([[embeddings[image_path]
                                   for image_path in class_paths]
                                  for class_paths in path_array])
      if self._verbose:
        tf.logging.info(embedding_array.shape)
      return embedding_array, label_array, path_array

    output_list = tf.py_func(_build_one_instance_py, [],
                             [tf.float32, tf.int32, tf.string])
    instance_input, instance_output, instance_info = output_list
    instance_input = tf.nn.l2_normalize(instance_input, axis=-1)
    instance_info = tf.regex_replace(instance_info, "\x00*", "")

    if self._verbose:
      tf.logging.info("input_batch: {} ".format(instance_input.shape))
      tf.logging.info("output_batch: {} ".format(instance_output.shape))
      tf.logging.info("info_batch: {} ".format(instance_info.shape))

    split_sizes = [tr_size, val_size]
    tr_input, val_input = tf.split(instance_input, split_sizes, axis=1)
    tr_output, val_output = tf.split(instance_output, split_sizes, axis=1)
    tr_info, val_info = tf.split(instance_info, split_sizes, axis=1)
    if self._verbose:
      tf.logging.info("tr_output: {} ".format(tr_output))
      tf.logging.info("val_output: {}".format(val_output))

    with tf.control_dependencies(
        self._check_labels(num_classes, tr_size, val_size,
                           tr_output, val_output)):
      tr_output = tf.identity(tr_output)
      val_output = tf.identity(val_output)

    return tr_input, tr_output, tr_info, val_input, val_output, val_info

  def get_batch(self, batch_size, num_classes, tr_size, val_size,
                num_threads=10):
    """Returns a batch of random N-way K-shot classification problem instances.

    Args:
      batch_size: number of problem instances in the batch.
      num_classes: N in N-way classification.
      tr_size: K in K-shot; number of training examples per class.
      val_size: number of validation examples per class.
      num_threads: number of threads used to sample problem instances in
      parallel.

    Returns:
      A ProblemInstance of Tensors with the following shapes:
      - tr_input: (batch_size, num_classes, tr_size, NDIM): training image
      embeddings.
      - tr_output: (batch_size, num_classes, tr_size, 1): training image
      labels.
      - tr_info: (batch_size, num_classes, tr_size): training image file
      names.
      - val_input: (batch_size, num_classes, val_size, NDIM): validation
      image embeddings.
      - val_output: (batch_size, num_classes, val_size, 1): validation
      image labels.
      - val_info: (batch_size, num_classes, val_size): validation image
      file names.
    """
    if self._verbose:
      num_threads = 1
    one_instance = self.get_instance(num_classes, tr_size, val_size)

    tr_data_size = (num_classes, tr_size)
    val_data_size = (num_classes, val_size)
    task_batch = tf.train.shuffle_batch(one_instance, batch_size=batch_size,
                                        capacity=1000, min_after_dequeue=0,
                                        enqueue_many=False,
                                        shapes=[tr_data_size + (NDIM,),
                                                tr_data_size + (1,),
                                                tr_data_size,
                                                val_data_size + (NDIM,),
                                                val_data_size + (1,),
                                                val_data_size],
                                        num_threads=num_threads)

    if self._verbose:
      tf.logging.info(task_batch)

    return ProblemInstance(*task_batch)

  def _check_labels(self, num_classes, tr_size, val_size,
                    tr_output, val_output):
    correct_label_sum = (num_classes*(num_classes-1))//2
    tr_label_sum = tf.reduce_sum(tr_output)/tr_size
    val_label_sum = tf.reduce_sum(val_output)/val_size
    all_label_asserts = [
        tf.assert_equal(tf.to_int32(tr_label_sum), correct_label_sum),
        tf.assert_equal(tf.to_int32(val_label_sum), correct_label_sum),
    ]
    return all_label_asserts
