from abc import ABCMeta, abstractmethod, abstractproperty
import torch.utils.data as data
from src.utils.vocabulary import Vocabulary

TRAIN_ONLY_ERR_MSG = "{} only supported for train dataset! Instead saw {}"

class AbstractDataset(data.Dataset):
    __metaclass__ = ABCMeta

    class Utterance:
        ## minimum elements all datasets must have; id, length, tokens
        def __init__(self, tokens):
            # Initialization for dummy utterance
            self.id = None
            self.label = 0
            self.speaker = None
            # TODO: clean text before processing
            self.tokens = tokens
            self.length = len(self.tokens)

    def __init__(self):
        self.name = ""
        self.total_length = 0
        self.vocabulary = Vocabulary()
        self.train_dataset = []
        self.test_dataset = []
        self.valid_dataset = []
        self.label_set_size = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample

    def __add__(self, other):
        ## length combines
        aggregated_dataset = AbstractDataset()

        aggregated_dataset.total_length = self.total_length + other.total_length
        aggregated_dataset.label_set_size = max(self.label_set_size, other.label_set_size)
        aggregated_dataset.vocabulary = self.vocabulary + other.vocabulary
        aggregated_dataset.train_dataset = self.train_dataset + other.train_dataset
        aggregated_dataset.test_dataset = self.test_dataset + other.test_dataset
        aggregated_dataset.valid_dataset = self.valid_dataset + other.valid_dataset

        return aggregated_dataset

    def get_full_dataset(self):
        return self.train_dataset + self.valid_dataset + self.test_dataset