"""
This file contains the hyper parameters for the model.

You can change the values to see how they affect the training.
"""
import torch
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.35
vocab_size = 279 #enc.n_vocab 
# ------------
