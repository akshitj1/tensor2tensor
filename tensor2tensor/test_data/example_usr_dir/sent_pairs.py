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
"""Data generators for MultiNLI (https://www.nyu.edu/projects/bowman/multinli/).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import pickle as pk
import os.path as osp
import tensorflow as tf

EOS = text_encoder.EOS_ID


class MultinliProblem(problem.Problem):
    """Base class for MultiNLI classification problems."""

    @property
    def num_shards(self):
        return 10

    @property
    def vocab_file(self):
        return 'sim_pair.vocab'

    @property
    def targeted_vocab_size(self):
        return 2**10

    @property
    def _matched(self):
        raise NotImplementedError()

    _LABELS = {'contradict', 'identity'}

    # @property
    # def _train_file(self):
    #     return 'multinli_1.0/multinli_1.0_train.jsonl'
    #
    # @property
    # def _dev_file(self):
    #     if self._matched:
    #         return 'multinli_1.0/multinli_1.0_dev_matched.jsonl'
    #     else:
    #         return 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl'


    def _examples(self, data_dir, tmp_dir, train):
        # del data_dir
        # base_dir='/mnt/data/abhishek.y/Datasets/t2t/t2t_data'
        base_dir='/Users/abhishek.y/Documents/Datasets/t2t/t2t/t2t_data'
        file_name = osp.join(base_dir,'sim_pairs_train.pk') if train else osp.join(base_dir,'sim_pairs_test.pk')
        with open(file=file_name, mode='rb') as f:
            examples=pk.load(f)
        return examples

    def _inputs_and_targets(self, encoder, examples):
        for e in examples:
            enc_s1 = encoder.encode(e['sentence1'])
            enc_s2 = encoder.encode(e['sentence2'])

            yield {
                'input_x': enc_s1 + [EOS],
                'input_y': enc_s2 + [EOS],
                'targets': [e['label']]
            }

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        train_paths = self.training_filepaths(
            data_dir, self.num_shards, shuffled=False)
        dev_paths = self.dev_filepaths(data_dir, 1, shuffled=False)

        train_examples = self._examples(data_dir, tmp_dir, train=True)
        dev_examples = self._examples(data_dir, tmp_dir, train=False)

        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_file, self.targeted_vocab_size,
            (e['sentence1'] + ' ' + e['sentence2']
             for e in train_examples + dev_examples)
        )

        generator_utils.generate_dataset_and_shuffle(
            self._inputs_and_targets(encoder, train_examples), train_paths,
            self._inputs_and_targets(encoder, dev_examples), dev_paths)

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        source_vocab_size = self._encoders['input_x'].vocab_size
        p.input_modality = {
            'input_x': (registry.Modalities.SYMBOL, source_vocab_size),
            'input_y': (registry.Modalities.SYMBOL, source_vocab_size)
        }
        p.target_modality = (registry.Modalities.CLASS_LABEL, 2)
        p.input_space_id = problem.SpaceID.EN_TOK
        p.target_space_id = problem.SpaceID.GENERIC

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_file)
        encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        return {
            'input_x': encoder,
            'input_y': encoder,
            'targets': text_encoder.ClassLabelEncoder(self._LABELS),
        }

    def example_reading_spec(self):
        data_fields = {
            'input_x': tf.VarLenFeature(tf.int64),
            'input_y': tf.VarLenFeature(tf.int64),
            'targets': tf.FixedLenFeature([1], tf.int64),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def eval_metrics(self):
        return [metrics.Metrics.ACC]


@registry.register_problem
class SimPairs(MultinliProblem):
    """MultiNLI with matched dev set."""

    @property
    def _matched(self):
        return True
