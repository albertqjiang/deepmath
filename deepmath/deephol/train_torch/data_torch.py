import os
import torch
from deepmath.deephol.train.data import pairwise_thm_parser
from deepmath.deephol.train_torch.extractor_torch import TacticExtractor
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


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



