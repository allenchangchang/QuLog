import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import argparse


class QLSTM(nn.Module):
    # use 'qiskit.ibmq' instead to run on hardware
    
    class SelectiveMeasure(tq.QuantumModule):
        def __init__(self, n_qubits):
            super().__init__()
            self.n_qubits = n_qubits
            
            if n_qubits <= 4:
                self.measure = tq.MeasureAll(tq.PauliZ)
            else:
                self.measure = tq.MeasureMultipleTimes([
                    {
                        'wires': list(range(4)),  # 只测量前4个量子比特
                        'observables': ['z'] * 4
                    }
                ])

        def forward(self, qdev: tq.QuantumDevice):
            return self.measure(qdev)
    
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits=4):
            # output :
            # batch size, seq len, embed dim
            super().__init__()    
            self.n_wires = n_qubits
            f_list = []
            # 在此处修改量子线路数
            # 对量子线路的初始方法在 func 中，这里与后面 V 操作不一样
            for i in range(self.n_wires):
                f_list.append({'input_idx': [i], 'func': 'rx', 'wires': [i]})
            
            # encoder 对应量子线路的 U 操作
            self.encoder = tq.GeneralEncoder(f_list)
            # self.encoder = tq.StateEncoder()
            # funcs = ['rx', 'rx', 'rx', 'rx']
            # self.encoder = tq.MultiPhaseEncoder(funcs)
            
            # 下面对应论文的 V 操作
            # 对应量子线率的V操作
            # 此处每一个rx或者ry是一个量子操作，
            # 相当于定义了一种传统机器学习中的模型
            # 此处统一使用rx的标识符，
            # 但是注意：后面的旋转函数才是其真实的旋转方法
            # 在此处即可修改量子线路
            for i in range(self.n_wires):
                setattr(self, f'rx{i}', tq.RY(has_params=True, trainable=True))
                setattr(self, f'rx{i + self.n_wires}', tq.RX(has_params=True, trainable=True))
                
            # for i in range(self.n_wires):
            #     setattr(self, f'rx{i+self.n_wires}', tq.RY(has_params=True, trainable=True))
            
        #     self.encoder = tq.GeneralEncoder(
        # [   {'input_idx': [0], 'func': 'rx', 'wires': [0]},
        #     {'input_idx': [1], 'func': 'rx', 'wires': [1]},
        #     {'input_idx': [2], 'func': 'rx', 'wires': [2]},
        #     {'input_idx': [3], 'func': 'rx', 'wires': [3]},
        # ])
        #     self.rx0 = tq.RX(has_params=True, trainable=True)
        #     self.rx1 = tq.RX(has_params=True, trainable=True)
        #     self.rx2 = tq.RX(has_params=True, trainable=True)
        #     self.rx3 = tq.RX(has_params=True, trainable=True)
            # 使用选择性测量
            self.measure = tq.MeasureAll(tq.PauliZ)
            
        def U_block(self, inputs):
            '''
                TODO:
                    实现一个U(\theta)
                
                RX -> CNOT -> RY -> CNOT -> RX -> CNOT -> RX -> CNOT
                
                wires
                
                lstm_1 = LSTM(RX, RY, CNOT)
                lstm_2 = LSTM()
                x -> lstm_1 -> lstm_2
                
            '''
            pass
            
        def forward(self, x):
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            
            # 无论是哪个旋转操作，forward都是一样的方法。
            for i in range(self.n_wires):
                rx_func = getattr(self, f'rx{i}')  # 动态获取属性
                rx_func(qdev, wires=i)  # 调用对应的 旋转 函数
                rx_func = getattr(self, f'rx{i + self.n_wires}')  # 动态获取属性
                rx_func(qdev, wires=i)  # 调用对应的 旋转 函数
                
            
            # self.rx0(qdev, wires=0)
            # self.rx1(qdev, wires=1)
            # self.rx2(qdev, wires=2)
            # self.rx3(qdev, wires=3)
            for k in range(self.n_wires):
                if k==self.n_wires-1:
                    tqf.cnot(qdev, wires=[k, 0]) 
                else:
                    tqf.cnot(qdev, wires=[k, k+1])
            return(self.measure(qdev))
    
    class QLayer_forget(QLayer):
        pass
        
    class QLayer_input(QLayer):
        pass
             
    class QLayer_update(QLayer):
        pass
               
    class QLayer_output(QLayer):
        pass
        
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=4,
                n_qlayers=1,
                batch_first=True,
                return_sequences=False, 
                return_state=False,
                backend="qiskit.ibmq"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget':self.QLayer_forget(n_qubits=self.n_qubits),
            'input': self.QLayer_input(n_qubits=self.n_qubits),
            'update':self.QLayer_update(n_qubits=self.n_qubits),
            'output':self.QLayer_output(n_qubits=self.n_qubits)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        #self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
    
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
        def __init__(self, n_qubits=8):
            super().__init__()    
            self.n_wires = n_qubits
            f_list = []
            for i in range(self.n_wires):
                f_list.append({'input_idx': [i], 'func': 'rx', 'wires': [i]})
            self.encoder = tq.GeneralEncoder(f_list)
            for i in range(self.n_wires):
                setattr(self, f'rx{i}', tq.RX(has_params=True, trainable=True))
            
        #     self.n_wires = 8
        #     self.encoder = tq.GeneralEncoder(
        # [   {'input_idx': [0], 'func': 'rx', 'wires': [0]},
        #     {'input_idx': [1], 'func': 'rx', 'wires': [1]},
        #     {'input_idx': [2], 'func': 'rx', 'wires': [2]},
        #     {'input_idx': [3], 'func': 'rx', 'wires': [2]},
        #     {'input_idx': [4], 'func': 'rx', 'wires': [4]},
        #     {'input_idx': [5], 'func': 'rx', 'wires': [5]},
        #     {'input_idx': [6], 'func': 'rx', 'wires': [6]},
        #     {'input_idx': [7], 'func': 'rx', 'wires': [7]},
        # ])
        #     self.rx0 = tq.RX(has_params=True, trainable=True)
        #     self.rx1 = tq.RX(has_params=True, trainable=True)
        #     self.rx2 = tq.RX(has_params=True, trainable=True)
        #     self.rx3 = tq.RX(has_params=True, trainable=True)
        #     self.rx4 = tq.RX(has_params=True, trainable=True)
        #     self.rx5 = tq.RX(has_params=True, trainable=True)
        #     self.rx6 = tq.RX(has_params=True, trainable=True)
        #     self.rx7 = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward (self, x, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            for i in range(self.n_wires):
                rx_func = getattr(self, f'rx{i}')  # 动态获取属性
                rx_func(self.q_device, wires=i)  # 调用对应的 rx 函数
            
            # self.rx0(self.q_device, wires=0)
            # self.rx1(self.q_device, wires=1)
            # self.rx2(self.q_device, wires=2)
            # self.rx3(self.q_device, wires=3)
            # self.rx4(self.q_device, wires=4)
            # self.rx5(self.q_device, wires=5)
            # self.rx6(self.q_device, wires=6)
            # self.rx7(self.q_device, wires=7)
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
        # assert n_qubits == embed_dim, "Number of qubits ({n_qubits}) does not match embedding dim ({embed_dim})"
        
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.dev = tq.QuantumDevice(self.n_qubits, bsz=batch_size)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear_in = nn.Linear(embed_dim, n_qubits)
        self.linear_out = nn.Linear(n_qubits, embed_dim)
        
    def forward(self, x, mask=None):
        # x = self.linear_in(x)
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"
        self.dev.reset_states(batch_size)
        # print(x.size())

        K = [self.q_layer(x[:, t, :].clone(),self.dev) for t in range(seq_len)]
        Q = [self.q_layer(x[:, t, :].clone(),self.dev) for t in range(seq_len)]
        V = [self.q_layer(x[:, t, :].clone(),self.dev) for t in range(seq_len)]
        K = torch.Tensor(pad_sequence(K))
        Q = torch.Tensor(pad_sequence(Q))
        V = torch.Tensor(pad_sequence(V))
        x = self.downstream(Q, K, V, batch_size, mask)
        output = [self.q_layer(x[:, t, :],self.dev) for t in range(seq_len)]
        output = torch.Tensor(pad_sequence(output)).clone()
        
        # output = self.linear_out(output)
        return output
    
    
class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForwardBase, self).__init__()
        self.linear_in = nn.Linear(embed_dim, ffn_dim)
        self.linear_out = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    #def forward(self, x):
     #   raise NotImplementedError("Base class does not implement forward function")


class FeedForwardQuantum(FeedForwardBase):
    class QL(tq.QuantumModule):
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
                    
    def __init__(self, embed_dim, n_qubits, n_qlayers=1, dropout=0.1, q_device="qiskit.ibmq"):
        
        super(FeedForwardQuantum, self).__init__(embed_dim, ffn_dim=n_qubits, dropout=dropout)
        
        # assert n_qubits == embed_dim, "Number of qubits ({n_qubits}) does not match embedding dim ({embed_dim})"
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        # self.dev = tq.QuantumDevice(self.n_qubits)
        # 在forward中初始化dev
        self.dev = None
        self.q_l = self.QL()
        
        self.linear_in = nn.Linear(embed_dim, n_qubits)
        self.linear_out = nn.Linear(n_qubits, embed_dim)
    
    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.dev = qdev
        batch_size, seq_len, _ = x.size()
        x = self.linear_in(x)
        o = [self.q_l(x[:, t, :].detach(),self.dev) for t in range(seq_len)]
        x = torch.Tensor(pad_sequence(o))
        x = self.linear_out(x)
        return x
    

class TransformerBlockBase(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_head: int,
                 ff_dim: int,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlockBase, self).__init__()
        self.attn = None
        self.ffn = None
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.attn(x)
        print(f"attn_output size : {attn_output}")
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)

        return x
    

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 n_qlayers: int = 1,
                 dropout: float = 0.1,
                 batch_size: int = 1024,
                 mask=None,
                 q_device='qiskit.ibmq'):
        super(TransformerBlockQuantum, self).__init__(embed_dim, num_heads, ffn_dim, dropout, mask)
        
        self.n_qubits_transformer = n_qubits_transformer
        self.n_qubits_ffn = n_qubits_ffn
        self.n_qlayers = n_qlayers

        self.attn = MultiHeadAttentionQuantum(embed_dim,
                                              num_heads,
                                              n_qubits=n_qubits_transformer,
                                              n_qlayers=n_qlayers,
                                              dropout=dropout,
                                              mask=mask,
                                              batch_size=batch_size,
                                              q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, n_qubits_ffn, n_qlayers, q_device=q_device)
        else:
            raise ValueError("n_qubits_ffn <= 0")


def test():
    embed = torch.randn((4096, 10, 300))
    # out = embed
    lin = nn.Linear(300, 8)
    embed = lin(embed)
    attention = MultiHeadAttentionQuantum(embed_dim=8, 
                                        num_heads=1, 
                                        n_qubits=8,
                                        n_qlayers=1,
                                        batch_size=4096,
                                        q_device="cpu")
    transformer = TransformerBlockQuantum(embed_dim=8,
                                        num_heads=1,
                                        n_qubits_transformer=8,
                                        n_qlayers=1,
                                        batch_size=4096,
                                        q_device="cpu")
    out = attention(embed)
    out = nn.Linear(8, 300)(out)
    print(out.size())
    
    out = transformer(embed)
    print(out.size())

    
if __name__=="__main__":
    test()