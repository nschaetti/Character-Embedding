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
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from modules import LanguageModel
from torch.utils.data import DataLoader
import datasets
import numpy as np


####################################################
# Main function
####################################################


# Argument parser
parser = argparse.ArgumentParser(description="Character embedding extraction")

# Argument
parser.add_argument("--dim", type=int, help="Embedding dimension")
parser.add_argument("--n-gram", type=int, help="N-gram model", default=1)
parser.add_argument("--context-size", type=int, help="Content size", default=1)
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
parser.add_argument("--output", type=str, help="Embedding output file", default='char_embedding.p')
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--batch-size", type=int, help="Batch size", default=10)
parser.add_argument("--max-sample-size", type=int, help="Maximum sample size", default=40000)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Init random seed
torch.manual_seed(1)

# Wikipedia character dataset
wiki_dataset = datasets.WikipediaCharacter(context_size=args.context_size, n_gram=args.n_gram)

# Token to ix and voc size
_, voc_size = wiki_dataset.token_to_ix_voc_size()

# Dataset loader
wiki_dataset_loader = DataLoader(wiki_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=datasets.WikipediaCharacter.collate)

# Embedding layer
embedding_layer = nn.Embedding(voc_size, args.dim)

# Counter list
counter_list = list()

# Losses
losses = []

# Objective function
loss_function = nn.NLLLoss()

# Our model
model = LanguageModel(voc_size, args.dim, args.context_size)
if args.cuda:
    model.cuda()
# end if

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

# For each epoch
for epoch in range(args.epoch):

    # Total loss
    if args.cuda:
        total_loss = torch.cuda.FloatTensor([0])
    else:
        total_loss = torch.FloatTensor([0])
    # end if

    # Print dataset
    for index, data in enumerate(wiki_dataset_loader):
        # Data
        sample_inputs, sample_outputs = data

        # Sample size
        sample_size = sample_inputs.size(0)

        # For each samples
        for i in np.arange(0, sample_size, args.max_sample_size):
            # Sample
            inputs, outputs = sample_inputs[i:i+args.max_sample_size], sample_outputs[i:i+args.max_sample_size]

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if args.cuda:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            # end if

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

            # Print if first
            if epoch == 0 and index == 0 and i == 0:
                print(u"Starting loss {}".format(loss.data[0]))
            # end if
    # end for

    # Print
    print(u"Epoch {}, loss {}".format(total_loss[0]))
# end for

# Save
torch.save(model.embeddings.weight, open(args.output, 'wb'))
