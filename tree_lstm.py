import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import get_torch_variable_from_np, get_data


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, deprel_vocab_size):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

        self.lx = nn.Linear(self.in_dim, self.mem_dim)
        self.lh = nn.Linear(self.mem_dim, self.mem_dim)
        self.lb = nn.Parameter(torch.ones(deprel_vocab_size, self.mem_dim))

    def node_forward(self, inputs, child_c, child_h, deprels):

        if deprels is not None:
            l = F.sigmoid(
                self.lh(child_h) +
                self.lx(inputs).repeat(len(child_h), 1) +  
                self.lb[deprels]
            )

            child_h = torch.mul(l, child_h)

            # child_c = torch.mul(l, child_c)
        
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)

        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)

        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        
        return c, h

    def forward(self, tree, inputs): #, tree_hidden
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs) #, tree_hidden

        if tree.num_children == 0:
            child_c = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            deprels = None
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            deprels = get_torch_variable_from_np(np.array([c.deprel for c in tree.children]))

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h, deprels)

        # tree_hidden[tree.idx] = tree.state

        return tree.state