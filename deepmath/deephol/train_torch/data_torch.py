import os
import torch
from deepmath.deephol.train_torch.extractor_torch import TacticExtractor
import tensorflow as tf

tf.compat.v1.enable_eager_execution()


def generic_parser(serialized_example, feature_list, label_list):
    """Parses a HOL example, keeping requested features and labels.

    Args:
      serialized_example: A tf.Example for a parameterized tactic application.
      feature_list: List of string feature names to parse (subset of features).
      label_list: List of string label names to parse (subset of labels).

    Returns:
      features, labels: dicts with keys of feature_list, label_list respectively.
    """
    example = tf.parse_single_example(
        serialized_example,
        features={
            # Subgoal features
            # goal: the consequent term of the subgoal as a string.
            'goal': tf.FixedLenFeature((), tf.string, default_value=''),
            # goal_asl: list of hypotheses of the subgoal.
            'goal_asl': tf.VarLenFeature(dtype=tf.string),
            # Parameterized tactic applied to the subgoal
            # tactic: string name of tactic that is applied to this subgoal.
            'tactic': tf.FixedLenFeature((), tf.string, default_value=''),
            # tac_id: integer id of tactic.
            'tac_id': tf.FixedLenFeature((), tf.int64, default_value=-1),
            # thms: list of tactic arguments of type thm.
            'thms': tf.VarLenFeature(dtype=tf.string),
            # thms_hard_negatives: list of hard negative theorem parameter
            # arguments
            'thms_hard_negatives': tf.VarLenFeature(dtype=tf.string),
        })

    for key in ('goal_asl', 'thms', 'thms_hard_negatives'):
        if key in example:
            example[key] = tf.sparse_tensor_to_dense(example[key], default_value='')

    features = {key: example[key] for key in feature_list}
    labels = {key: example[key] for key in label_list}
    return features, labels


def _choose_one_theorem_at_random(thms):
    """Adds tf ops to pick one theorem at random from a list of theorems."""
    size_of_thms = tf.size(thms)

    def get_an_element():
        random_index = tf.random_uniform([],
                                         minval=0,
                                         maxval=size_of_thms,
                                         dtype=tf.int32)
        return thms[random_index]

    return tf.cond(size_of_thms > 0, get_an_element, lambda: '')


def pairwise_thm_parser(serialized_example, source, params):
    """Strips out a tactic id, goal term string, and random thm parameter.

    Args:
      serialized_example: A tf.Example for a parameterized tactic application.
      source: source of the example.
      params: Hyperparameters for the input.

    Returns:
      features['goal']: a string of the goal term.
      features['thms']: a string of a randomly chosen thm parameter or empty str.
      features['thms_hard_negatives']: list of strings, each a hard negative.
        Size controlled via params.
      labels['tac_id']: integer id of tactic applied.
    """
    del source  # unused

    feature_list = ['goal', 'thms', 'thms_hard_negatives']
    label_list = ['tac_id']
    features, labels = generic_parser(
        serialized_example, feature_list=feature_list, label_list=label_list)

    # thms: pick one uniformily at random
    features['thms'] = _choose_one_theorem_at_random(features['thms'])

    return features, labels


class DirectoryDataset(torch.utils.data.Dataset):
    def __init__(self, directory, prefix, params):
        self.examples = list()
        self.params = params
        for f_name in os.listdir(directory):
            if f_name.startswith(prefix):
                record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(directory, f_name))
                self.examples.extend(list(record_iterator))

    def __getitem__(self, item_index):
        features, labels = pairwise_thm_parser(self.examples[item_index], None, self.params)
        return features, labels

    def __len__(self):
        return len(self.examples)


class TacticDataset(torch.utils.data.Dataset):
    def __init__(self, directory, prefix, params):
        self.examples = list()
        self.params = params
        for f_name in os.listdir(directory):
            if f_name.startswith(prefix):
                record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(directory, f_name))
                self.examples.extend(list(record_iterator))
        self.extractor = TacticExtractor(params)
        self.extr = self.extractor.get_extractor()

    def get_multiple(self, batch_indices):
        return list(map(self.__getitem__, batch_indices))

    def __getitem__(self, item_index):
        features, labels = pairwise_thm_parser(self.examples[item_index], None, self.params)
        features, labels = self.extr(features, labels)
        return features, labels

    def __len__(self):
        return len(self.examples)


def get_directory_random_sampler(dataset):
    return torch.utils.data.RandomSampler(dataset)


def get_directory_batch_sampler(dataset, batch_size, drop_last=False):
    return torch.utils.data.BatchSampler(
        sampler=get_directory_random_sampler(dataset),
        batch_size=batch_size,
        drop_last=drop_last,
    )
