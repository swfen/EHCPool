from typing import Callable, Optional, Union
from torch import Tensor
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max
import torch.nn as nn
from torch_geometric.utils import softmax
import  numpy as np
from net.inits import uniform
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, zeros


def Iterative_topn(node_prem_lab, edge_perm, subgraph_num, retain_subnode, edge_index, batch_size, node_batch):

    if isinstance(subgraph_num, float):
        subgraph_num = subgraph_num * len(node_batch)/batch_size
    if isinstance(retain_subnode, float):
        retain_subnode = retain_subnode * len(node_batch)/batch_size

    arr = np.arange(0, subgraph_num*retain_subnode, dtype=int)
    node_prem_lab = node_prem_lab[:,arr]
    node_perm = node_prem_lab.view(-1)
    edge_perm_index = torch.index_select(edge_index, 1, edge_perm)

    edge_indices  = edge_perm_index[1,:]
    subgraph_edge_index_perm = torch.tensor((),dtype=torch.long,device=node_perm.device)
    sub_x_index = []
    x_perm = []
    for i in range(batch_size):
        node_prem_batch = node_prem_lab[i,:]
        ten = torch.tensor([],dtype=torch.long,device=node_perm.device)
        for j in range(subgraph_num):
            core = torch.tensor([node_prem_batch[j]],dtype=torch.long,device=node_perm.device)
            sub_x_index.append(core)
            mask = torch.eq(edge_indices, core)
            node_ne_lab = edge_perm_index[:,mask]
            a = torch.isin(node_ne_lab[0, :], ten)
            b = ~torch.isin(node_ne_lab[0,:], ten)
            node_ne_lab = node_ne_lab[:,~torch.isin(node_ne_lab[0,:], ten)]

            retain_arr = np.arange(0,retain_subnode-1,dtype=int)
            node_ne_lab = node_ne_lab[:, retain_arr]
            subgraph_edge_index_perm = torch.cat((subgraph_edge_index_perm,node_ne_lab),1)
            node_ne_lab = node_ne_lab[0,:]
            x_perm.append(torch.cat((core, node_ne_lab), 0))
            ten = torch.cat((node_ne_lab,ten),0)
            node_prem_batch = node_prem_batch[~torch.isin(node_prem_batch, ten)]

    x_perm_lab = torch.stack(x_perm)
    x_perm = x_perm_lab.view(-1)
    sub_core_x_index = torch.squeeze(torch.stack(sub_x_index),dim=1)


    return x_perm,x_perm_lab, sub_core_x_index, subgraph_edge_index_perm



def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        prem_lab = perm
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm,prem_lab

def get_node_score(edge_score,edge_index):
    device = torch.device('cuda:0')
    node_score = []
    edge_indices  = edge_index[1,:]

    for i in range(0,torch.max(edge_index).item()+1):
        mask = torch.eq(edge_indices, i)
        if mask == torch.Size([]):
            i = i+1
        else:
            node_score.append(torch.sum(edge_score[mask]))
    node_score = torch.stack(node_score).view(-1).to(device)
    return node_score

class Iterative_hard_clustering_pool(MessagePassing):
    r"""

    Args:
        node_channels (int): Size of each input node sample.
        edge_channels (int): Size of each input edge sample.
        subgraph_num (int): Graph pooling subgraph_num
        min_score (float, optional): Minimal node score，which is used to compute indices of pooled nodes
        node_ignorance （float）： Percentage of neglect of node features
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """

    def __init__(self, node_channels: int,
                 edge_channels: int,
                 subgraph_num: Union[int, float] = 0.1,
                 retain_subnode : Union[int, float] = 0.05,
                 batch_size: int = 1,
                 min_score: Optional[float] = None,
                 node_ignorance = 0,
                 multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.subgraph_num = subgraph_num
        self.retain_subnode  = retain_subnode
        self.batch_size = batch_size
        self.min_score = min_score
        self.node_ignorance = node_ignorance

        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight_node = Parameter(torch.Tensor(1, node_channels), requires_grad=True)
        self.weight_edge = Parameter(torch.Tensor(1, node_channels),requires_grad= True)
        self.Lin = nn.Sequential(nn.Linear(edge_channels, int(node_channels * node_channels))),

        self.reset_parameters()

    def reset_parameters(self):
        size = self.node_channels
        reset(self.Lin)
        uniform(size, self.weight_node)
        uniform(size, self.weight_edge)

    def forward(self, x, edge_index, edge_attr, node_batch=None, edge_batch=None, attn=None):
        """ Edge to node"""
        device = torch.device('cuda:0')
        if edge_batch is None:
            edge_subject_size = len(edge_attr)/self.batch_size
            edge_batch = []
            for i in range(self.batch_size):
                esub_size = torch.linspace(i, i, int(edge_subject_size), dtype= torch.long)
                edge_batch.append(esub_size)
        edge_batch = torch.stack(edge_batch).view(-1).to(device)

        if node_batch is None:
            node_batch = edge_index.new_zeros(x.size(0))

        attn = edge_attr if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        edge_score = (attn * self.weight_edge).sum(dim=-1)

        if self.min_score is None:
            edge_score = self.nonlinearity(edge_score / self.weight_edge.norm(p=2, dim=-1))
        else:
            edge_score = softmax(edge_score, edge_batch)

        edge_perm,edge_prem_lab = topk(edge_score, len(edge_score), edge_batch, self.min_score)
        edge_attr = edge_attr[edge_perm] * edge_score[edge_perm].view(-1, 1)
        edge_attr = self.multiplier * edge_attr if self.multiplier != 1 else edge_attr

        node_score = (self.nonlinearity((1-self.node_ignorance)*(x * self.weight_node).sum(dim=-1)/ self.weight_edge.norm(p=2, dim=-1)))+get_node_score(edge_score, edge_index)
        node_perm, node_prem_lab = topk(node_score, len(node_score), node_batch, self.min_score)

        """ Iterative_topn"""
        x_perm,x_perm_lab, sub_core_x_index, subgraph_edge_index_perm = Iterative_topn(node_prem_lab, edge_perm, self.subgraph_num, self.retain_subnode , edge_index, self.batch_size, node_batch)

        node_batch = node_batch[x_perm]
        x = x[x_perm,:]
        mask = []
        for i in range(len(subgraph_edge_index_perm[0,:])):
            a1 = torch.eq(edge_index[0, :], subgraph_edge_index_perm[0][i])
            tend = edge_index[:,a1]
            col1 = torch.nonzero(a1)[0]
            a2 = torch.eq(tend[1,:],subgraph_edge_index_perm[1][i])
            col2 = torch.nonzero(a2)[0]
            col = col1+col2
            mask.append(col)
        edge_index_len = len(subgraph_edge_index_perm[0, :])
        subgraph_edge_index_col1 = torch.arange(0,edge_index_len,dtype=torch.long,device=x.device)
        subgraph_edge_index_col2 = []
        maxrange = int(edge_index_len + edge_index_len / (self.retain_subnode - 1))
        for i in range(edge_index_len,maxrange):
            subgraph_edge_index_col2.append(torch.repeat_interleave(torch.tensor(i,dtype=torch.long,device=x.device), self.retain_subnode-1))

        subgraph_edge_index_col2 = torch.stack(subgraph_edge_index_col2).view(-1)
        subgraph_edge_index = torch.stack((subgraph_edge_index_col1,subgraph_edge_index_col2),dim=0)
        mask = torch.stack(mask).view(-1)
        edge_attr = edge_attr[mask,:]
        x_perm_lab = x_perm_lab.cpu().numpy()
        for batch_num in range(self.batch_size):
            for i in range(batch_num*self.subgraph_num,batch_num*self.subgraph_num+self.subgraph_num):
                for j in range(self.retain_subnode):
                    x_perm_lab[i][j] = x_perm_lab[i][j]-90*batch_num

        """ N-E aggregation"""
        x = self.propagate(subgraph_edge_index, x=x, edge_attr=edge_attr, size= None).to(x.device)
        core_node_index = torch.arange(0, len(node_batch), self.retain_subnode, dtype=torch.long, device=x.device)
        core_x = x[core_node_index, :]
        return core_x,x_perm_lab, subgraph_edge_index_perm,subgraph_edge_index, edge_attr,node_batch

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        Linea = self.Lin[0].to(x_j.device)
        weight = Linea(edge_attr)
        weight = weight.view(-1, self.edge_channels, self.node_channels)
        sum = torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
        return sum

    def __repr__(self) -> str:
        if self.min_score is None:
            subgraph_num = f'subgraph_num={self.subgraph_num}'
        else:
            subgraph_num = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.node_channels}, {subgraph_num}, '
                f'multiplier={self.multiplier})')
