from abc import ABCMeta, abstractmethod, abstractproperty
import torch.utils.data as data
from embed.utils.vocabulary import Vocabulary

TRAIN_ONLY_ERR_MSG = "{} only supported for train dataset! Instead saw {}"

class AbstractDataset(data.Dataset):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.total_length = 0
        self.vocabulary = Vocabulary()
        self.train_dataset = []
        self.test_dataset = []
        self.valid_dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample

    def __add__(self, other):
        ## length combines
        aggregated_dataset = AbstractDataset()

        aggregated_dataset.total_length = self.total_length + other.total_length
        aggregated_dataset.vocabulary = self.vocabulary + other.vocabulary
        aggregated_dataset.train_dataset = self.train_dataset + other.train_dataset
        aggregated_dataset.test_dataset = self.test_dataset + other.test_dataset
        aggregated_dataset.valid_dataset = self.valid_dataset + other.valid_dataset

        return aggregated_dataset

