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
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Settings
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'x', 'y', 'z', '.', ',', ';', ':', '-', '!', '?', '"', '\'', '(', ')', ' ', '0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9']

# Argument parser
parser = argparse.ArgumentParser(description="Character embedding visualization")

# Argument
parser.add_argument("--input", type=str, help="Embedding input file")
parser.add_argument("--output", type=str, help="Output image file")
parser.add_argument("--image-size", type=int, help="Image size", default=4000)
args = parser.parse_args()

# Load
token_to_ix, weights = torch.load(open(args.input, 'rb'))

# Embedding layer
embedding = nn.Embedding(weights.size(0), weights.size(1))
embedding.weight = nn.Parameter(weights)

# Embedding vectors
embedding_vectors = weights.numpy()

# T-SNE
tsne_embedding = TSNE(n_components=2).fit_transform(embedding_vectors)

# Select only needed vectors
idxs = [token_to_ix[c] for c in alphabet]
selected_vectors = tsne_embedding[idxs]

# Sub plt
fig, ax = plt.subplots()
ax.scatter(selected_vectors[:, 0], selected_vectors[:, 1])

# Show char
for c in alphabet:
    idx = token_to_ix[c]
    ax.annotate(c, (tsne_embedding[idx, 0], tsne_embedding[idx, 1]))
# end for

# Save
fig.savefig(args.output)

