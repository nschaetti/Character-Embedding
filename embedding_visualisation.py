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
parser = argparse.ArgumentParser(description="Character embedding visualization")

# Argument
parser.add_argument("--input", type=str, help="Embedding input file")
parser.add_argument("--output", type=str, help="Output image file")
parser.add_argument("--image-size", type=int, help="Image size", default=4000)
args = parser.parse_args()

# Load
token_to_ix, weights = torch.load(open(args.input, 'rb'))

# Embedding layer
weights = weights.cpu()
print(type(weights))
embedding = nn.Embedding(weights.data.size(0), weights.data.size(1))
embedding.weight = nn.Parameter(weights.data)

# Keys
print(token_to_ix.keys())

# Some vectors
print(embedding(torch.LongTensor([token_to_ix['a']])))
print(embedding(torch.LongTensor([token_to_ix['b']])))

