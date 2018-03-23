#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch.nn as nn
import torch.nn.functional as F


# Deep Language model
class DeepLanguageModel(nn.Module):

    # Constructor
    def __init__(self, vocab_size, embedding_dim, context_size, out_channel, linear_features=128):
        super(DeepLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channel)
        self.linear1 = nn.Linear(context_size * 2 * embedding_dim, linear_features)
        self.linear2 = nn.Linear(linear_features, vocab_size)
    # end __init__

    # Forward
    def forward(self, inputs):
        batch_size = inputs.size(0)
        embeds = self.embeddings(inputs)
        x = self.conv(embeds)
        x = x.view((batch_size, -1))
        print(x.size())
        exit()
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        # log_probs = F.softmax(out, dim=1)
        return log_probs
    # end forward

# end DeepLanguageModel
