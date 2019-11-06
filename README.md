# 10K-word-embeddings
Word embeddings learned from 10-K documents as decribed in paper:  
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3480902

# Getting Started
These instructions will help you get started with using the word embeddings in your projects.

# Prerequisities
Install PyTorch and Numpy

# Details
10k_word_embeddings.tar - Tar file containing word embeddings. Each embedding is of 300 dimension.  
vocab_to_int.tar- Tar file containing vocab to integer mapping. Vocab size is 159647 words.

# Usage

import torch  
embed = torch.load('10k_word_embeddings.tar')  
vocab_to_int = torch.load('vocab_to_int.tar')  

- **Use learned embeddings as pre-trained embeddings in a neural network:**  
from torch import nn  
embeddings = nn.Embedding(embed.shape[0], embed.shape[1])  
embeddings.weight.data.copy_(torch.from_numpy(embed))  
embeddings.weight.requires_grad = False  

- **Get learned embeddings of a word from vocabulary:**  
embeddings_profit = embed[vocab_to_int['profit']]
