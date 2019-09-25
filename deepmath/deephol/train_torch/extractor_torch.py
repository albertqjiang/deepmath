"""Extractor for HOLparam models. Tokenizes goals and theorems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from deepmath.deephol.train_torch import utils_torch
from torch_geometric.data import Data

import tensorflow as tf
import torch
tf.enable_eager_execution()


class TacticExtractor(object):
    """Extract terms/thms and tokenize based on vocab.

    Attributes:
      params: Hyperparameters.
      goal_table: Lookup table for goal vocab embeddings.
      thms_table: Lookup table for theorem parameter vocab embeddings.
      add_negative: Integer multiple ratio of negative examples to positives.
      all_thms: List of all training thms as strings.
      random_negatives: A batch of negative thm examples.
      goal_closed_negative_iterator: A iterator for negative goal closed examples.
    """

    def __init__(self, params):
        """Inits Extractor class with hyperparameters."""
        self.params = params

        # Create vocab lookup tables from existing vocab id lists.
        dataset_dir = params['dataset_dir']
        goal_file = os.path.join(dataset_dir, params['goal_vocab'])
        self.goal_table = utils_torch.vocab_table_from_file(goal_file)
        self.goal_table[r'('] = self.goal_table.get(r'(', len(self.goal_table))
        self.goal_table[r')'] = self.goal_table.get(r')', len(self.goal_table))
        if params['thm_vocab'] is not None:
            thms_file = os.path.join(dataset_dir, params['thm_vocab'])
            self.thms_table = utils_torch.vocab_table_from_file(thms_file)
        else:
            self.thms_table = self.goal_table
        self.thms_table[r'('] = self.thms_table.get(r'(', len(self.thms_table))
        self.thms_table[r')'] = self.thms_table.get(r')', len(self.thms_table))

        # Some vocab that appears are not in the vocab file
        missing_vocab_file = os.path.join(dataset_dir, params['missing_vocab'])
        self.missing_table = utils_torch.vocab_table_from_file(missing_vocab_file)
        for key in self.missing_table:
            if key not in self.goal_table:
                index = len(self.goal_table)
                self.goal_table[key] = index
            if key not in self.thms_table:
                index = len(self.thms_table)
                self.thms_table[key] = index

    def ast_to_geometric(self, ast, table):
        if isinstance(ast, list) and len(ast) == 1:
            ast = ast[0]
        node_index, connections, nodes = utils_torch.ast_to_connections(ast)
        if len(ast) > 0:
            assert node_index == 0
        else:
            return None
        for node in nodes:
            assert isinstance(node, str)
            if node not in table:
                self.missing_vocab.add(node)
        node_values = list(map(table.get, nodes))
        # print(node_values)
        node_value_tensor = torch.LongTensor(node_values)
        node_value_tensor = torch.reshape(node_value_tensor, [-1, 1])
        edge_index = torch.LongTensor(connections)
        edge_index = torch.transpose(edge_index, 0, 1)
        return Data(edge_index=edge_index, x=node_value_tensor)

    def tokenize_single_tm(self, single_tm, table):
        # TODO: original implementation truncates words that are too long
        # words = tf.slice(words, [0, 0],
        #                  [tf.shape(words)[0], self.params.truncate_size])
        ast = utils_torch.parse(single_tm)
        return self.ast_to_geometric(ast, table)

    def tokenize(self, tm, table):
        """Tokenizes tensor string according to lookup table."""
        tm_numpy = tm.numpy()
        if isinstance(tm_numpy, bytes):
            single_tm = tm_numpy.decode('utf-8')
            single_tm = single_tm.replace(r"(", r"( ")
            single_tm = single_tm.replace(r")", r" )")
            return self.tokenize_single_tm(single_tm, table)

        elif isinstance(tm_numpy, np.ndarray):
            all_tm_sparse = list()
            for single_tm in tm_numpy:
                single_tm = single_tm.decode('utf-8')
                single_tm = single_tm.replace(r"(", r"( ")
                single_tm = single_tm.replace(r")", r" )")
                all_tm_sparse.append(self.tokenize_single_tm(single_tm, table))
            return all_tm_sparse

    def get_extractor(self):
        """Returns extractor function based on initialized params."""

        def extractor(features, labels):
            """Converts 'goal' features and 'thms' labels to list of ids by vocab."""

            if 'goal' not in features:
                raise ValueError('goal feature missing.')
            if 'tac_id' not in labels:
                raise ValueError('tac_id label missing.')

            labels['tac_id'] = torch.LongTensor(tf.reshape(labels['tac_id'], [-1]).numpy())

            # tokenize 'goal' and 'thms'.
            tf.add_to_collection('goal_string', features['goal'])
            features['goal_ids'] = self.tokenize(features['goal'], self.goal_table)
            del features['goal']
            if 'thms' in features:
                del features['thms']
                del features['thms_hard_negatives']

            return features, labels

        return extractor
