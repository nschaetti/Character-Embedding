#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import os
from torch.utils.data import Dataset
import codecs
import torch
import numpy as np


# Wikipedia Character prediction
class WikipediaCharacter(Dataset):
    """
    Wikipedia Character prediction
    """

    # Constructor
    def __init__(self, context_size, token_to_ix, root='./data'):
        """
        Constructor
        :param context_size:
        :param token_to_ix:
        """
        # Properties
        self.root = root
        self.context_size = context_size
        self.token_to_ix = token_to_ix

        # Load file list
        self.files = self._load()
    # end __init__

    ############################################
    # PUBLIC
    ############################################

    # Collate
    def collate(self, batch):
        """
        Collate
        :param batch:
        :return:
        """
        print(batch.size())
        exit()
    # end collate

    ############################################
    # OVERRIDE
    ############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.files)
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        # Path to text
        path_to_text = os.path.join(self.root, self.files[item])

        # Get text
        text = codecs.open(path_to_text, 'rb', encoding='utf-8').read()

        # Text length
        text_length = len(text)
        sample_length = text_length - self.context_size

        # Inputs and output
        inputs = torch.LongTensor(sample_length, self.context_size)
        outputs = torch.LongTensor(sample_length)

        # Build tuple with (preceding chars, target char)
        for i in np.arange(self.context_size, text_length):
            pos = 0
            for j in np.arange(i - self.context_size, i):
                inputs[i-self.context_size, pos] = self.token_to_ix[text[j]]
                pos += 1
            # end for
            outputs[i-self.context_size] = self.token_to_ix[text[i]]
        # end for

        return inputs, outputs
    # end __getitem__

    ############################################
    # PRIVATE
    ############################################

    # Load file list
    def _load(self):
        """
        Load file list
        :return:
        """
        return os.listdir(self.root)
    # end _load

# end WikipediaCharacter
