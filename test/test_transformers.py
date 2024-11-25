import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchquantum as tq
import torchquantum.functional as tqf
import argparse
import tqdm
import time

import torch
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 mask=None,
                 use_bias=False):
        super(MultiHeadAttentionBase, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # projection dimensions
        self.k_linear = None
        self.q_linear = None
        self.v_linear = None
        self.combine_heads = None
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
    
    def separate_heads(self, x):
        '''
        split into N heads
        from (batch_size, seq_len, embed_dim)
        to   (batch_size, seq_len, num_heads, embed_dim)
        then transpose (1,2) to (batch_size, num_heads, seq_len, embed_dim)
        to make mat mult straightforward for each head
        '''
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def attention(self, query, key, value, mask=None, dropout=None):
        '''
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k))V
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # see also: https://tensorchiefs.github.io/dlday2018/tutorial/einsum.html
        #scores = torch.einsum('bijh, bkjh -> bikh', query, key) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        attn = torch.matmul(scores, value)
        return attn, scores
    
    def downstream(self, query, key, value, batch_size, mask=None):
        Q = self.separate_heads(query)
        K = self.separate_heads(key)
        V = self.separate_heads(value)

        x, self.attn_weights = self.attention(Q, K, V, mask, dropout=self.dropout)

        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return concat
        # output = self.combine_heads(concat)
        # return output

   # def forward(self, x, mask=None):
    #    raise NotImplementedError("Base class does not execute forward function.")
        
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False):
        super(MultiHeadAttentionClassical, self).__init__(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, mask=mask, use_bias=use_bias)

        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = self.k_linear(x)
        Q = self.q_linear(x)
        V = self.v_linear(x)

        x = self.downstream(Q, K, V, batch_size, mask)
        output = self.combine_heads(x)
        return output


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()    
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
        [   {'input_idx': [0], 'func': 'rx', 'wires': [0]},
            {'input_idx': [1], 'func': 'rx', 'wires': [1]},
            {'input_idx': [2], 'func': 'rx', 'wires': [2]},
            {'input_idx': [3], 'func': 'rx', 'wires': [2]},
            {'input_idx': [4], 'func': 'rx', 'wires': [4]},
            {'input_idx': [5], 'func': 'rx', 'wires': [5]},
             {'input_idx': [6], 'func': 'rx', 'wires': [6]},
          {'input_idx': [7], 'func': 'rx', 'wires': [7]},
        ])
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.rx3 = tq.RX(has_params=True, trainable=True)
            self.rx4 = tq.RX(has_params=True, trainable=True)
            self.rx5 = tq.RX(has_params=True, trainable=True)
            self.rx6 = tq.RX(has_params=True, trainable=True)
            self.rx7 = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward (self, x, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            self.rx0(self.q_device, wires=0)
            self.rx1(self.q_device, wires=1)
            self.rx2(self.q_device, wires=2)
            self.rx3(self.q_device, wires=3)
            self.rx4(self.q_device, wires=4)
            self.rx5(self.q_device, wires=5)
            self.rx6(self.q_device, wires=6)
            self.rx7(self.q_device, wires=7)
            for k in range(self.n_wires):
                if k==self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[k, 0]) 
                else:
                    tqf.cnot(self.q_device, wires=[k, k+1])
            return(self.measure(self.q_device))
            
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False,
                 n_qubits: int = 4,
                 n_qlayers: int = 1,
                 batch_size: int = 1024,
                 q_device="default.qubit"):
        super(MultiHeadAttentionQuantum, self).__init__(embed_dim, num_heads, dropout=dropout, mask=mask, use_bias=use_bias)
        
        # todo: add intermediate layer to "dress" quantum circuit
        assert n_qubits == embed_dim, "Number of qubits ({n_qubits}) does not match embedding dim ({embed_dim})"
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.dev = tq.QuantumDevice(self.n_qubits, bsz=batch_size)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = [self.q_layer(x[:, t, :].clone(),self.dev) for t in range(seq_len)]
        Q = [self.q_layer(x[:, t, :].clone(),self.dev) for t in range(seq_len)]
        V = [self.q_layer(x[:, t, :].clone(),self.dev) for t in range(seq_len)]
        K = torch.Tensor(pad_sequence(K))
        Q = torch.Tensor(pad_sequence(Q))
        V = torch.Tensor(pad_sequence(V))
        x = self.downstream(Q, K, V, batch_size, mask)
        output = [self.q_layer(x[:, t, :],self.dev) for t in range(seq_len)]
        output = torch.Tensor(pad_sequence(output)).clone()
        return output

    
if __name__ == '__main__':
    bs = 7
    inputs = torch.randn((32, 10, bs))
    atten_block = MultiHeadAttentionQuantum(embed_dim=bs, num_heads=1, n_qubits=bs, batch_size=32)
    outputs = atten_block(inputs)
    print(outputs.size())