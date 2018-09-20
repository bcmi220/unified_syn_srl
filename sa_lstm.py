import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import get_torch_variable_from_np, get_data


class SyntaxAwareLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, deprel_vocab_size):
        super(SyntaxAwareLSTM, self).__init__()

        self.in_dim = in_dim

        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 5 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 5 * self.mem_dim)

        self.lx = nn.Linear(self.in_dim, self.mem_dim)
        self.lh = nn.Linear(self.mem_dim, self.mem_dim)
        self.lb = nn.Parameter(torch.ones(deprel_vocab_size, self.mem_dim))

    def node_forward(self, inputs, ht_1, ct_1, relative, deprels):

        ious = self.ioux(inputs) + self.iouh(ht_1)

        # input gate; output gate; forget gate; syntactic gate;

        i, o, f, s, u = torch.split(ious, ious.size(1) // 5, dim=1)

        i, o, f, s, u = F.sigmoid(i), F.sigmoid(o), F.sigmoid(s), F.sigmoid(f), F.tanh(u)

        c = torch.mul(f, ct_1) + torch.mul(i, u)

        h = torch.mul(o, F.tanh(c))

        if len(relative) > 0:

            rel_var = torch.cat(relative, dim=0)

            l = F.sigmoid(
                self.lh(rel_var) +
                self.lx(inputs).repeat(len(rel_var), 1) + 
                self.lb[deprels]
            )

            rel_var = torch.mul(l, rel_var)
   
            h = h + torch.mul(s, F.tanh(torch.sum(rel_var, dim=0, keepdim=True)))

        return c, h

    def forward(self, inputs, relative, deprels):
        seq_len = inputs.shape[0]
        fw_hidden = [None for _ in range(seq_len)]
        fw_context = [None for _ in range(seq_len)]
        bw_hidden = [None for _ in range(seq_len)]
        bw_context = [None for _ in range(seq_len)]
        hidden = []

        init_c = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        init_h = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))

        # forward and backward
        for idx in range(seq_len):
            if idx == 0:
                c, h = self.node_forward(inputs[idx], init_h, init_c, [], None)
            else:
                if len(relative[idx, 0]) > 0:
                    rel_hidden = [fw_hidden[ind] for ind in relative[idx, 0]]
                    deprel_hidden = get_torch_variable_from_np(np.array(deprels[idx, 0]))
                else:
                    rel_hidden = []
                    deprel_hidden = None
                c, h = self.node_forward(inputs[idx], fw_hidden[idx - 1], fw_context[idx - 1],
                                     rel_hidden, deprel_hidden)
            fw_hidden[idx] = h
            fw_context[idx] = c

            ridx = seq_len - idx - 1
            if ridx == inputs.shape[0] - 1:
                c, h = self.node_forward(inputs[ridx], init_c, init_h, [], None)
            else:
                if len(relative[ridx, 1]) > 0:
                    rel_hidden = [bw_hidden[ind] for ind in relative[ridx, 1]]
                    deprel_hidden = get_torch_variable_from_np(np.array(deprels[ridx, 1]))
                else:
                    rel_hidden = []
                    deprel_hidden = None
                c, h = self.node_forward(inputs[ridx], bw_hidden[ridx + 1], bw_context[ridx + 1],
                                     rel_hidden, deprel_hidden)
            bw_hidden[ridx] = h
            bw_context[ridx] = c

        for idx in range(seq_len):
            hidden.append(torch.cat([fw_hidden[idx], bw_hidden[idx]], dim=1))

        hidden = torch.cat(hidden, dim=0)

        return hidden
