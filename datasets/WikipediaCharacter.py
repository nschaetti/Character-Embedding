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
    def __init__(self, context_size, root='./data', token_to_ix=None, n_gram=1):
        """
        Constructor
        :param context_size:
        :param token_to_ix:
        """
        # Properties
        self.root = root
        self.context_size = context_size
        self.token_to_ix = token_to_ix
        self.n_gram = n_gram

        # Load file list
        self.files = self._load()
    # end __init__

    ############################################
    # PUBLIC
    ############################################

    # Compute token to ix and voc size
    def token_to_ix_voc_size(self):
        """
        Compute token to ix and voc size
        :param dataset_path:
        :return:
        """
        token_to_ix = {}
        index = 0
        # For each file
        for file_name in os.listdir(self.root):
            text_data = codecs.open(os.path.join(self.root, file_name), 'r', encoding='utf-8').read()
            for i in range(len(text_data)):
                character = text_data[i:i+self.n_gram]
                if character not in token_to_ix:
                    token_to_ix[character] = index
                    index += 1
                # end if
            # end for
        # end for
        self.token_to_ix = token_to_ix
        return token_to_ix, index
    # end token_to_ix_voc_size

    ############################################
    # STATIC
    ############################################

    # Collate
    @staticmethod
    def collate(batch):
        """
        Collate
        :param batch:
        :return:
        """
        # Inputs and outputs
        inputs = torch.LongTensor()
        outputs = torch.LongTensor()

        # For each batch
        for i in range(len(batch)):
            if i == 0:
                inputs = batch[i][0]
                outputs = batch[i][1]
            else:
                inputs = torch.cat((inputs, batch[i][0]), dim=0)
                outputs = torch.cat((outputs, batch[i][1]), dim=0)
            # end if
        # end for
        return inputs, outputs
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
        sample_length = text_length - self.context_size * self.n_gram * 2 - self.n_gram + 1

        # Inputs and output
        inputs = torch.LongTensor(sample_length, self.context_size * 2)
        outputs = torch.LongTensor(sample_length)

        # Start and end
        start = self.context_size * self.n_gram
        end = start + sample_length

        # Build tuple with (preceding chars, target char)
        sample_pos = 0
        for i in np.arange(start, end):
            # Before
            pos = 0
            for j in np.arange(i - self.context_size * self.n_gram, i, self.n_gram):
                current_gram = text[j:j+self.n_gram]
                inputs[sample_pos, pos] = self.token_to_ix[current_gram]
                pos += 1
            # end for

            # After
            pos = self.context_size
            for j in np.arange(i + self.n_gram, i + self.n_gram + self.context_size * self.n_gram, self.n_gram):
                current_gram = text[j:j + self.n_gram]
                inputs[sample_pos, pos] = self.token_to_ix[current_gram]
                pos += 1
            # end for

            # Current target gram
            target_gram = text[i:i+self.n_gram]
            print(u"{} : {}".format(target_gram, self.token_to_ix[target_gram]))

            # Target output
            outputs[sample_pos] = self.token_to_ix[target_gram]

            # Sample pos
            sample_pos += 1
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
