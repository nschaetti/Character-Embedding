#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch.nn as nn
import torch.nn.functional as F


# Deep Language model
class DeepLanguageModel(nn.Module):

    # Constructor
    def __init__(self, vocab_size, embedding_dim, context_size, out_channel, kernel_size, stride):
        # Super
        super(DeepLanguageModel, self).__init__()

        # Linear features
        self.linear_features = int(vocab_size / 10.0)

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channel, kernel_size=kernel_size, stride=stride)

        # Linear layers
        self.linear1 = nn.Linear(context_size * 2 * embedding_dim, self.linear_features)
        self.linear2 = nn.Linear(self.linear_features, vocab_size)
    # end __init__

    # Forward
    def forward(self, inputs):
        """
        Forward
        :param inputs:
        :return:
        """
        # Batch size
        batch_size = inputs.size(0)
        print(batch_size)
        # Embedding layer
        embeds = self.embeddings(inputs)
        print(embeds.size())
        exit()
        # Convolutional layer
        x = self.conv(embeds)
        x = x.view((batch_size, -1))
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        # log_probs = F.softmax(out, dim=1)
        return log_probs
    # end forward

# end DeepLanguageModel
