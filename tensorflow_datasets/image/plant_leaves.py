# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""Healhy and unhealthy plant leaves dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@misc{,
  author={Siddharth Singh Chouhan, Ajay Kaul, Uday Pratap Singh, Sanjeev
Jain},
  title={A Database of Leaf Images: Practice towards Plant Conservation with
Plant Pathology},
  howpublished={Mendeley Data},
  year={2019}
}
"""

_DESCRIPTION = """
This dataset consists of 4502 images of healthy and unhealthy plant leaves
divided into 22 categories by species and state of health. The images are in
high resolution JPG format.

Dataset URL: https://data.mendeley.com/datasets/hb74ynkjcn/1
License: http://creativecommons.org/licenses/by/4.0
"""

# File name prefix to label mapping.
_LABEL_MAPPING = [
    ("0014", "Alstonia Scholaris (P2) diseased"),
    ("0003", "Alstonia Scholaris (P2) healthy"),
    ("0013", "Arjun (P1) diseased"),
    ("0002", "Arjun (P1) healthy"),
    ("0016", "Bael (P4) diseased"),
    ("0008", "Basil (P8) healthy"),
    ("0022", "Chinar (P11) diseased"),
    ("0011", "Chinar (P11) healthy"),
    ("0015", "Gauva (P3) diseased"),
    ("0004", "Gauva (P3) healthy"),
    ("0017", "Jamun (P5) diseased"),
    ("0005", "Jamun (P5) healthy"),
    ("0018", "Jatropha (P6) diseased"),
    ("0006", "Jatropha (P6) healthy"),
    ("0021", "Lemon (P10) diseased"),
    ("0010", "Lemon (P10) healthy"),
    ("0012", "Mango (P0) diseased"),
    ("0001", "Mango (P0) healthy"),
    ("0020", "Pomegranate (P9) diseased"),
    ("0009", "Pomegranate (P9) healthy"),
    ("0019", "Pongamia Pinnata (P7) diseased"),
    ("0007", "Pongamia Pinnata (P7) healthy"),
]
_URLS_FNAME = "image/plant_leaves_urls.txt"
_MAX_DOWNLOAD_RETRY = 10


class PlantLeaves(tfds.core.GeneratorBasedBuilder):
  """Healhy and unhealthy plant leaves dataset."""

  VERSION = tfds.core.Version("0.1.0")

  UNSTABLE = "Each image is a separate download. Some might fail."

  def _info(self):
    labels = list(zip(*_LABEL_MAPPING))[1]
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "image/filename": tfds.features.Text(),
            "label": tfds.features.ClassLabel(names=labels)
        }),
        supervised_keys=("image", "label"),
        urls=["https://data.mendeley.com/datasets/hb74ynkjcn/1"],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # Batch download for this dataset is broken, therefore images have to be
    # downloaded independently from a list of urls.
    with tf.io.gfile.GFile(tfds.core.get_tfds_path(_URLS_FNAME)) as f:
      name_to_url_map = {
          os.path.basename(l.strip()): l.strip() for l in f.readlines()
      }
      retry_count = 0
      image_files = {}
      # We have to retry due to rare 504 HTTP errors. Downloads are cached,
      # therefore this shouldn't cause successful downloads to be retried.
      while True:
        try:
          image_files = dl_manager.download(name_to_url_map)
          break
        except:
          ++retry_count
          if retry_count == _MAX_DOWNLOAD_RETRY:
            raise
      return [
          tfds.core.SplitGenerator(
              name=tfds.Split.TRAIN,
              num_shards=1,
              gen_kwargs={"image_files": image_files})
      ]

  def _generate_examples(self, image_files):
    """Yields examples."""
    label_map = {pattern: label for pattern, label in _LABEL_MAPPING}
    regexp = re.compile("^(\d\d\d\d)_.*\.JPG$")
    # Assigns labels to images based on label mapping.
    for original_fname, fpath in image_files.items():
      match = regexp.match(original_fname)
      if match and match.group(1) in label_map:
        label = label_map[match.group(1)]
        record = {
            "image": fpath,
            "image/filename": original_fname,
            "label": label,
        }
        yield "{}/{}".format(label, original_fname), record
