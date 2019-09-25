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


class Extractor(object):
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
        if params['thm_vocab'] is not None:
            thms_file = os.path.join(dataset_dir, params['thm_vocab'])
            self.thms_table = utils_torch.vocab_table_from_file(thms_file)
        else:
            self.thms_table = self.goal_table
        missing_vocab_file = os.path.join(dataset_dir, params['missing_vocab'])
        self.missing_table = utils_torch.vocab_table_from_file(missing_vocab_file)
        for key in self.missing_table:
            if key not in self.goal_table:
                index = len(self.goal_table)
                self.goal_table[key] = index
            if key not in self.thms_table:
                index = len(self.thms_table)
                self.thms_table[key] = index

        # If adding negative examples, create a list of all training thms.
        self.add_negative = params['ratio_neg_examples']
        if self.add_negative:
            # Path to negative examples text file.
            # File should contain one example per line, in the same format as the
            # training examples.
            all_thms_file = os.path.join(dataset_dir, 'thms_ls.train')
            # Get a constant batch_size tensor of tokenized random train theorems.
            d = tf.data.TextLineDataset(all_thms_file)
            d = d.repeat()
            # Shuffle within a sliding window slightly larger than the set of thms.
            d = d.shuffle(
                buffer_size=params.negative_example_shuffle_buffer,
                reshuffle_each_iteration=True)
            d = d.batch(
                (self.params['ratio_neg_examples']) * self.params['batch_size'])
            d = d.make_one_shot_iterator()
            self.random_negatives = d.get_next()

    def tokenize_single_tm(self, single_tm, table):
        # TODO: original implementation truncates words that are too long
        # words = tf.slice(words, [0, 0],
        #                  [tf.shape(words)[0], self.params.truncate_size])
        words = single_tm.split()
        word_tokens = list(map(table.get, words))
        sparse_tensor_shape = torch.Size([len(table), len(word_tokens)])
        indices = torch.LongTensor([word_tokens, list(range(len(word_tokens)))])
        values = torch.ones([len(word_tokens)])
        words_sparse = torch.sparse.FloatTensor(indices, values, sparse_tensor_shape)
        return words_sparse

    def tokenize(self, tm, table):
        """Tokenizes tensor string according to lookup table."""
        tm = tf.strings.join(['<START> ', tf.strings.strip(tm), ' <END>'])
        # Remove parentheses - they can be recovered for S-expressions.
        tm = tf.strings.regex_replace(tm, r'\(', ' ')
        tm = tf.strings.regex_replace(tm, r'\)', ' ')
        tm_numpy = tm.numpy()
        if isinstance(tm_numpy, bytes):
            single_tm = tm_numpy.decode('utf-8')
            return self.tokenize_single_tm(single_tm, table)

        elif isinstance(tm_numpy, np.ndarray):
            all_tm_sparse = list()
            for single_tm in tm_numpy:
                single_tm = single_tm.decode('utf-8')
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

            # Tile the related features/labels (goals are tiled after embedding).
            goal_tiling_size = self.params.ratio_neg_examples + 1
            labels['tac_id'] = tf.reshape(labels['tac_id'], [-1])
            labels['tac_id'] = tf.tile(labels['tac_id'], [goal_tiling_size])
            labels['tac_present'] = tf.ones(
                [goal_tiling_size * self.params.batch_size])

            # Tokenize the thm parameter (assumes single thm in list)
            if 'thms' in features:
                if self.add_negative:
                    hard_negatives = features['thms_hard_negatives']
                    hard_negatives = tf.reshape(tf.transpose(hard_negatives), [-1])

                    def hard_or_random_picker(hard_random_pair):
                        hard, random = hard_random_pair
                        hard_not_present = tf.equal(hard, tf.constant('<NULL>'))
                        return tf.cond(hard_not_present, lambda: random, lambda: hard)

                    neg_thms = tf.map_fn(
                        hard_or_random_picker, (hard_negatives, self.random_negatives),
                        dtype=tf.string)
                    features['thms'] = tf.reshape(features['thms'], [-1])
                    labels['thm_label'] = tf.concat([
                        tf.ones(tf.shape(features['thms'])[0], dtype=tf.int32),
                        tf.zeros(tf.shape(neg_thms)[0], dtype=tf.int32)
                    ],
                        axis=0)
                    features['thms'] = tf.concat([features['thms'], neg_thms], axis=0)

            if labels is not None and 'thms' in labels:
                labels['thm_ids'] = self.tokenize(labels['thms'], self.thms_table)
                del labels['thms']

            # tokenize 'goal' and 'thms'.
            tf.add_to_collection('goal_string', features['goal'])
            features['goal_ids'] = self.tokenize(features['goal'], self.goal_table)
            del features['goal']
            if 'thms' in features:
                tf.add_to_collection('thm_string', features['thms'])
                features['thm_ids'] = self.tokenize(features['thms'], self.thms_table)
                del features['thms']
                del features['thms_hard_negatives']

            return features, labels

        return extractor


class ASTExtractor(object):
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

        # If adding negative examples, create a list of all training thms.
        self.add_negative = params['ratio_neg_examples']
        if self.add_negative:
            # Path to negative examples text file.
            # File should contain one example per line, in the same format as the
            # training examples.
            all_thms_file = os.path.join(dataset_dir, 'thms_ls.train')
            # Get a constant batch_size tensor of tokenized random train theorems.
            d = tf.data.TextLineDataset(all_thms_file)
            d = d.repeat()
            # Shuffle within a sliding window slightly larger than the set of thms.
            d = d.shuffle(
                buffer_size=params.negative_example_shuffle_buffer,
                reshuffle_each_iteration=True)
            d = d.batch(
                (self.params['ratio_neg_examples']) * self.params['batch_size'])
            d = d.make_one_shot_iterator()
            self.random_negatives = d.get_next()

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
        # for node in nodes:
        #     assert isinstance(node, str)
        #     if node not in table:
        #         print(node)
        #         raise NotImplementedError
        node_values = list(map(table.get, nodes))
        # print(node_values)
        node_value_tensor = torch.FloatTensor(node_values)
        node_value_tensor = torch.reshape(node_value_tensor, [-1, 1])
        edge_index = torch.LongTensor(connections)
        edge_index = torch.transpose(edge_index, 0, 1)
        return Data(edge_index=edge_index, x=node_value_tensor)

    def tokenize_single_tm(self, single_tm, table):
        # TODO: original implementation truncates words that are too long
        # words = tf.slice(words, [0, 0],
        #                  [tf.shape(words)[0], self.params.truncate_size])
        # words = single_tm.split()
        # for word in words:
        #     if word not in table:
        #         self.missing_tokens.add(word)
        # print(single_tm)

        ast = utils_torch.parse(single_tm)
        # print("The AST is", ast)
        # print(self.ast_to_geometric(ast, table))
        return self.ast_to_geometric(ast, table)

    def tokenize(self, tm, table):
        """Tokenizes tensor string according to lookup table."""
        # tm = tf.strings.join(['<START> ', tf.strings.strip(tm), ' <END>'])
        # Remove parentheses - they can be recovered for S-expressions.
        # tm = tf.strings.regex_replace(tm, r'\(', r'( ')
        # tm = tf.strings.regex_replace(tm, r'\)', r' )')

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

            # Tile the related features/labels (goals are tiled after embedding).
            goal_tiling_size = self.params.ratio_neg_examples + 1
            labels['tac_id'] = tf.reshape(labels['tac_id'], [-1])
            labels['tac_id'] = tf.tile(labels['tac_id'], [goal_tiling_size])
            labels['tac_present'] = tf.ones(
                [goal_tiling_size * self.params.batch_size])

            # Tokenize the thm parameter (assumes single thm in list)
            if 'thms' in features:
                if self.add_negative:
                    hard_negatives = features['thms_hard_negatives']
                    hard_negatives = tf.reshape(tf.transpose(hard_negatives), [-1])

                    def hard_or_random_picker(hard_random_pair):
                        hard, random = hard_random_pair
                        hard_not_present = tf.equal(hard, tf.constant('<NULL>'))
                        return tf.cond(hard_not_present, lambda: random, lambda: hard)

                    neg_thms = tf.map_fn(
                        hard_or_random_picker, (hard_negatives, self.random_negatives),
                        dtype=tf.string)
                    features['thms'] = tf.reshape(features['thms'], [-1])
                    labels['thm_label'] = tf.concat([
                        tf.ones(tf.shape(features['thms'])[0], dtype=tf.int32),
                        tf.zeros(tf.shape(neg_thms)[0], dtype=tf.int32)
                    ],
                        axis=0)
                    features['thms'] = tf.concat([features['thms'], neg_thms], axis=0)

            if labels is not None and 'thms' in labels:
                labels['thm_ids'] = self.tokenize(labels['thms'], self.thms_table)
                del labels['thms']

            # tokenize 'goal' and 'thms'.
            tf.add_to_collection('goal_string', features['goal'])
            features['goal_ids'] = self.tokenize(features['goal'], self.goal_table)
            del features['goal']
            if 'thms' in features:
                tf.add_to_collection('thm_string', features['thms'])
                features['thm_ids'] = self.tokenize(features['thms'], self.thms_table)
                del features['thms']
                del features['thms_hard_negatives']

            return features, labels

        return extractor


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

        # If adding negative examples, create a list of all training thms.
        self.add_negative = params['ratio_neg_examples']
        if self.add_negative:
            # Path to negative examples text file.
            # File should contain one example per line, in the same format as the
            # training examples.
            all_thms_file = os.path.join(dataset_dir, 'thms_ls.train')
            # Get a constant batch_size tensor of tokenized random train theorems.
            d = tf.data.TextLineDataset(all_thms_file)
            d = d.repeat()
            # Shuffle within a sliding window slightly larger than the set of thms.
            d = d.shuffle(
                buffer_size=params.negative_example_shuffle_buffer,
                reshuffle_each_iteration=True)
            d = d.batch(
                (self.params['ratio_neg_examples']) * self.params['batch_size'])
            d = d.make_one_shot_iterator()
            self.random_negatives = d.get_next()

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

        self.missing_vocab = set()

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
