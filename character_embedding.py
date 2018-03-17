#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import argparse
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import codecs
import numpy as np
from modules import LanguageModel
from torch.utils.data import DataLoader
import os
import datasets


####################################################
# Main function
####################################################


# Argument parser
parser = argparse.ArgumentParser(description="Character embedding extraction")

# Argument
parser.add_argument("--dataset", type=str, help="Input file")
parser.add_argument("--dim", type=int, help="Embedding dimension")
parser.add_argument("--n-gram", type=int, help="N-gram model")
parser.add_argument("--context-size", type=int, help="Content size")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
parser.add_argument("--output", type=str, help="Embedding output file", default='char_embedding.p')
args = parser.parse_args()

# Settings
batch_size = 64

# Init random seed
torch.manual_seed(1)

# Wikipedia character dataset
wiki_dataset = datasets.WikipediaCharacter(context_size=args.context_size, n_gram=args.n_gram)

# Token to ix and voc size
_, voc_size = wiki_dataset.token_to_ix_voc_size()

# Dataset loader
wiki_dataset_loader = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True, collate_fn=datasets.WikipediaCharacter.collate)

# Embedding layer
embedding_layer = nn.Embedding(voc_size, args.dim)

# Losses
losses = []

# Objective function
loss_function = nn.NLLLoss()

# Our model
model = LanguageModel(voc_size, args.dim, args.context_size)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Print dataset
for data in wiki_dataset_loader:
    # Data
    inputs, outputs = data
    print(inputs.size())
    print(outputs.size())
    print(inputs)
    print(outputs)
    exit()
# end for

# For each epoch
for epoch in range(args.epoch):
    total_loss = torch.Tensor([0])
    # Print dataset
    for data in wiki_dataset_loader:
        # Data
        inputs, outputs = data

        # To variable
        inputs, outputs = Variable(inputs), Variable(outputs)

        # Reset gradients
        model.zero_grad()

        # Forward pass
        log_probs = model(inputs)

        # Compute loss function
        loss = loss_function(log_probs, outputs)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Add total loss
        total_loss += loss.data
    # end for
    losses.append(total_loss)
    print(total_loss[0])
# end for

print(token_to_ix['a'])
print(model.embeddings(autograd.Variable(torch.LongTensor([token_to_ix['a']]))))

print(token_to_ix['b'])
print(model.embeddings(autograd.Variable(torch.LongTensor([token_to_ix['b']]))))

# Save
torch.save(model.embeddings, open(args.output, 'wb'))
