import sys
sys.path.append('./')

import torch
import torch.nn as nn
from torch.autograd import Variable
# from .Factory import QLSTM
from .Factory_v1 import QLSTM, MultiHeadAttentionQuantum, TransformerBlockQuantum


class quantumlstm(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 n_qubits,
                 n_qlayers,
                 batch_first,
                 return_sequences,
                 return_state,
                 vocab_size, 
                 num_layers,
                 embedding_dim):
        super(quantumlstm, self).__init__()
        print("calling QLSTM")
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm = QLSTM(self.embedding_dim,
                            hidden_size,
                            n_qubits,
                            n_qlayers,
                            batch_first,
                            return_sequences,
                            return_state,
                            backend="qiskit.ibmq"
                            )
        self.fc0 = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, device):
        input0 = features[0]
        embed0 = self.embedding(input0)
        # h0 = torch.zeros(self.num_layers, embed0.size(0),
        #                  self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, embed0.size(0),
        #                  self.hidden_size).to(device)
        # import pdb; pdb.set_trace()
        out, _ = self.lstm(embed0)
        out0 = self.fc0(out[:, -1, :])
        # out0 = out0.softmax(dim=-1)
        return out0, out0


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(self.embedding_dim,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=False)
        self.fc0 = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, device):
        input0 = features[0]
        embed0 = self.embedding(input0)
        # h0 = torch.zeros(self.num_layers, embed0.size(0),
        #                  self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, embed0.size(0),
        #                  self.hidden_size).to(device)
        out, _ = self.lstm(embed0)
        out0 = self.fc0(out[:, -1, :])
        # out0 = out0.softmax(dim=-1)
        return out0, out0


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim, num_keys=2):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.num_directions = 2
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        self.attention_size = self.hidden_size
        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))
        self.sequence_length = 100

    def attention_net(self, lstm_output, device):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device):
        inp = features[2]
        self.sequence_length = inp.size(1)
        out, _ = self.lstm(inp)
        out = self.attention_net(out, device)
        out = self.fc1(out)
        out = self.fc2(out)
        return out, out


class qrobustlog(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 n_qubits,
                 n_qlayers,
                 batch_first,
                 return_sequences,
                 return_state,
                 embedding_dim,
                 ):
        super(qrobustlog, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = QLSTM(input_size,
                            hidden_size,
                            n_qubits,
                            n_qlayers,
                            batch_first,
                            return_sequences,
                            return_state,
                            ) # 
        self.hidden_size = hidden_size
        self.n_qubits = 8
        self.num_directions = 2
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        
        # 添加全连接层，将 lstm 的输出维度转换为 n_qubits
        self.fc_transform = nn.Linear(hidden_size, self.n_qubits)
        # attention out 输出转换成原始维度
        self.fc_out = nn.Linear(self.n_qubits, hidden_size * self.num_directions)

         # 添加 MultiHeadAttentionQuantum 层
        self.attention = MultiHeadAttentionQuantum(embed_dim=self.n_qubits,
                                                   num_heads=2,
                                                   n_qubits=self.n_qubits,
                                                   n_qlayers=n_qlayers,
                                                   q_device="cuda")
        self.sequence_length = 100

    def forward(self, features, device):
        inp = features[2]
        self.sequence_length = inp.size(1)
        out, _ = self.lstm(inp)
        
        # 使用全连接层将 lstm 的输出维度转换为 n_qubits
        out = self.fc_transform(out)
        # 测试lstm的输出fc后维度
        # print(f'fc out: {out.size()}')
        out = self.attention(out)
        # print(f'attention out: {out.size()}')
        out = self.fc_out(out)
        
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out, out


#log key add embedding
class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size + 1, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm0 = nn.LSTM(1,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, features, device):
        # print(len(features), "fdsklfjsd")
        input0, input1 = features[1], features[2]    # quantitative parttern, Semantic partter
        # print(input1.shape)
        # embed0 = self.embedding(input0)

        h0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))

        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out, out

class qloganomaly(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers,
                 n_qubits,
                 n_qlayers,
                 batch_first,
                 return_sequences,
                 return_state,
                 vocab_size, 
                 embedding_dim):
        super(qloganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size + 1, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm0 = QLSTM(1,
                            hidden_size,
                            n_qubits,
                            n_qlayers,
                            batch_first,
                            return_sequences,
                            return_state,
                            )

        self.lstm1 = QLSTM(input_size,
                            hidden_size,
                            n_qubits,
                            n_qlayers,
                            batch_first,
                            return_sequences,
                            return_state,
                            )
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, features, device):
        # print(len(features), "fdsklfjsd")
        input0, input1 = features[1], features[2]
        # print(input1.shape)
        # embed0 = self.embedding(input0)

        h0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)
        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))

        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out, out

