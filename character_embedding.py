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
from modules import DeepLanguageModel, LanguageModel
from torch.utils.data import DataLoader
import datasets
import numpy as np
from decimal import Decimal
import torchlanguage.utils


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
parser.add_argument("--max-test-sample-size", type=int, help="Maximum test sample size", default=40000)
parser.add_argument("--n-linear", type=int, help="Linear size", default=40000)
parser.add_argument("--deep", action='store_true', help="Use deep model", default=False)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Init random seed
torch.manual_seed(1)

# Wikipedia character dataset
wiki_dataset = datasets.WikipediaCharacter(context_size=args.context_size, n_gram=args.n_gram)

# Token to ix and voc size
token_to_ix, voc_size = wiki_dataset.token_to_ix_voc_size()
print(u"Vocabulary size {}".format(voc_size))

# Dataset loader
wiki_dataset_loader = DataLoader(wiki_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=datasets.WikipediaCharacter.collate)

# Losses
losses = []

# Objective function
loss_function = nn.NLLLoss()

# Our model
if args.deep:
    model = DeepLanguageModel(voc_size, args.dim, args.context_size, out_channel=10, kernel_size=(20, 4), stride=(10, 0))
else:
    model = LanguageModel(voc_size, args.dim, args.context_size, args.n_linear)
# end if
if args.cuda:
    model.cuda()
# end if
best_perp = 2000000000

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

# For each epoch
for epoch in np.arange(-1, args.epoch, 1):
    # Log
    print(u"Epoch {}".format(epoch))

    # Total loss
    train_loss = 0.0
    train_total = 0.0
    test_loss = 0.0
    test_total = 0.0
    model.train()

    # Skip pre-epoch
    if epoch >= 0:
        # Train
        wiki_dataset_loader.dataset.set_train(True)

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
                train_loss += loss.data[0]
                train_total += 1.0
            # end for

            # Log
            if index % 1000 == 0:
                print(u"\tSample {}, loss {}".format(index, train_loss / train_total))
            # end if
        # end for
    # end if

    # Test
    wiki_dataset_loader.dataset.set_train(False)
    model.eval()

    # Print dataset
    p_sum = torch.FloatTensor([0.0]) if not args.cuda else torch.cuda.FloatTensor([0.0])
    p_total = 0.0
    for index, data in enumerate(wiki_dataset_loader):
        # Data
        sample_inputs, sample_outputs = data

        # Sample size
        sample_size = sample_inputs.size(0)

        # For each samples
        for i in np.arange(0, sample_size, args.max_test_sample_size):
            # Sample
            inputs, outputs = sample_inputs[i:i + args.max_test_sample_size], sample_outputs[i:i + args.max_test_sample_size]

            # To variable
            inputs, outputs = Variable(inputs), Variable(outputs)
            if args.cuda:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            # end if

            # Forward pass
            log_probs = model(inputs)

            # Compute loss function
            loss = loss_function(log_probs, outputs)

            # Perplexity
            p_sum += torchlanguage.utils.cumperplexity(log_probs.data, outputs.data, log=True)
            p_total += outputs.size(0)

            # Add total loss
            test_loss += loss.data[0]
            test_total += 1.0
        # end for

        # Perplexity
        if index % 1000 == 0:
            print(u"\tTest loss {}, perplexity {}".format(test_loss / test_total, np.power(2, -1.0 / p_total * float(p_sum[0]))))
        # end if
    # end for

    # Perplexity
    perplexity = np.power(2, -1.0 / p_total * float(p_sum[0]))

    # Print
    if epoch >= 0:
        print(u"Epoch {}, training loss {}, test loss {}, perplexity {}".format(epoch, train_loss / train_total,
                                                                                test_loss / test_total, perplexity))
    else:
        print(u"Epoch {}, test loss {}, perplexity {}".format(epoch, test_loss / test_total, perplexity))
    # end if

    # Save if better
    if perplexity < best_perp:
        print(u"Save best model to {}".format(args.output))
        torch.save((token_to_ix, torch.FloatTensor(model.embeddings.weight.data.cpu())), open(args.output, 'wb'))
        best_perp = perplexity
    # end if
# end for
