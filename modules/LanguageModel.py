#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch.nn as nn
import torch.nn.functional as F


# Language model
class LanguageModel(nn.Module):

    # Constructor
    def __init__(self, vocab_size, embedding_dim, context_size, linear_features=128):
        super(LanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * 2 * embedding_dim, linear_features)
        self.linear2 = nn.Linear(linear_features, vocab_size)
    # end __init__

    # Forward
    def forward(self, inputs):
        batch_size = inputs.size(0)
        embeds = self.embeddings(inputs)
        embeds = embeds.view((batch_size, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        # log_probs = F.softmax(out, dim=1)
        return log_probs
    # end forward

# end LanguageModel
