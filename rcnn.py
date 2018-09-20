import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import get_torch_variable_from_np, get_data


# module for RCNN
class RCNN(nn.Module):
    def __init__(self, in_dim, mem_dim, deprel_vocab_size):
        super(RCNN, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.deprel_emb = nn.Parameter(torch.ones(deprel_vocab_size, self.mem_dim))
        self.mlp = nn.Linear(in_dim+self.mem_dim*2, self.mem_dim)
        
    def node_forward(self, inputs, child_h, child_deprels):

        assert len(child_h) == len(child_deprels)

        node_repr = torch.cat([inputs.repeat(len(child_h), 1), child_deprels, child_h], dim=1)

        node_repr = F.tanh(self.mlp(node_repr))

        node_repr = node_repr.view(1,self.mem_dim,-1)

        node_repr = F.max_pool1d(node_repr, node_repr.shape[-1])

        return node_repr.view(1, -1)

    def forward(self, tree, inputs): #, tree_hidden
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs) #, tree_hidden

        if tree.num_children == 0:
            child_h = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_deprels = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_h = [x.state for x in tree.children]
            child_h = torch.cat(child_h, dim=0)
            child_deprels = self.deprel_emb[get_torch_variable_from_np(np.array([c.deprel for c in tree.children]))]

        tree.state = self.node_forward(inputs[tree.idx], child_h, child_deprels)

        return tree.state