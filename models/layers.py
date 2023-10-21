import pickle

import numpy as np
import torch
from torch import nn

from models.utils import SingleHeadAttentionLayer ,MultiHeadAttention
import torch.nn.functional as F
from models.utils import softAttention
class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))#疾病的embedding
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))#邻居的embedding
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))#不相关的embedding
        self.cate_embeddings=nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(157, 150)))#不相关的embedding

    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings,self.cate_embeddings

class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size,cate_adj):
        super().__init__()
        self.adj = adj

        #self.adj=nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(4880, 4880)))
        print(self.adj.shape)
        self.cate_adj=cate_adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()
        #self.code_tsne={}
    def forward(self, code_x, neighbor, c_embeddings, n_embeddings,cate_embeddings,cate_x):
        # code_x2=code_x.to('cpu')

        center_codes = torch.unsqueeze(code_x, dim=-1)#[1,4880]
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)#[1,4880]
        center_embeddings = center_codes * c_embeddings#mt*M,eq2的第一项#[4880,48]
        neighbor_embeddings = neighbor_codes * n_embeddings#eq3的第一项[4880,48]
        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)#local context[4880,48]
        cn_embeddings = center_codes * torch.matmul(self.adj, neighbor_embeddings)#diagnosis global context
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)#eq2的第二项
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)#eq3的第三项
        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings))#eq2本地疾病的embdding
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings))#eq3邻居的embedding
        cate_codes=torch.unsqueeze(cate_x,dim=-1)
        cate_embeddings = cate_codes * cate_embeddings
        #cate_embeddings=cate_codes*torch.matmul(self.cate_adj,cate_embeddings)
        #print("cate_embeddings",cate_embeddings.shape)
        # print(code_x2)
        # print(np.where(code_x2 == True))

        # for i in np.where(code_x2==True)[0]:
        #     if i not in self.code_tsne:
        #         self.code_tsne[i]=co_embeddings[i]
        #

        return co_embeddings, no_embeddings,cate_embeddings


class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size,dict,parent_dict):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        #self.gru=nn.LSTMCell(input_size=graph_size,hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        #self.single_head_attention=MultiHeadAttention(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.dict=dict
        self.parent=nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(20, 150)))
        self.dict2=parent_dict
        self.code_tsne = {}
        # print(self.dict2)
        # print(self.dict)
    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, cate_embedidngs,hidden_state=None):

        dict=self.dict
        dict2=self.dict2
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]#m1是persistent diseases
        m1_index = torch.where(m1 > 0)[0]#m2是emerging neighbors
        m2_index = torch.where(m2 > 0)[0]#m3是emerging unrelated diseases
        m3_index = torch.where(m3 > 0)[0]
        m1_index_cate=[]
        m2_index_cate=[]
        m3_index_cate=[]
        for i in m1_index:
            m1_index_cate.append(dict[int(i)])
        for j in m2_index:
            m2_index_cate.append(dict[int(j)])
        for k in m3_index:
            m3_index_cate.append(dict[int(k)])
        m1_index_cate_p = []
        m2_index_cate_p = []
        m3_index_cate_p = []
        for i in m1_index:
            m1_index_cate_p.append(dict2[int(i)])
        for j in m2_index:
            m2_index_cate_p.append(dict2[int(j)])
        for k in m3_index:
            m3_index_cate_p.append(dict2[int(k)])
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
        output_m1 = 0
        output_m23 = 0
        if len(m1_index) > 0:
            m1_embedding = co_embeddings[m1_index]#([9, 3 2])
            new_cate_embedidngs=cate_embedidngs[m1_index_cate]
            partent_cate_embeddings=self.parent[m1_index_cate_p]
            h = hidden_state[m1_index] if hidden_state is not None else None

            h_m1 = self.gru(m1_embedding, h)
            #h_new[m1_index] = h_m1+new_cate_embedidngs+partent_cate_embeddings
            #h_m1 = h_m1 + new_cate_embedidngs+partent_cate_embeddings#torch.Size([9, 150])


            #print(m1_index)
            #print(h_m1.shape)
            for index,i in enumerate(m1_index):
               # print(i)
                if int(i) not in self.code_tsne:
                    self.code_tsne[int(i)]=h_m1[index]

            output_m1, _ = torch.max(h_m1, dim=-2)
        if t > 0 and len(m2_index) + len(m3_index) > 0:

            q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])#[3,32]
            v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])#[3,32]
            h_m23 = self.activation(self.single_head_attention(q, q, v))#[3,150\
            # h_cate=torch.squeeze(cate_embedidngs[m2_index_cate+m3_index_cate])
            # h_cate2=torch.squeeze(self.parent[m2_index_cate_p+m3_index_cate_p])
            # h_m23=h_m23+h_cate+h_cate2
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]#[4880,150]
            h_cate=torch.squeeze(cate_embedidngs[m2_index_cate+m3_index_cate])
            h_cate2=torch.squeeze(self.parent[m2_index_cate_p+m3_index_cate_p])
            #h_m23=h_m23+h_cate+h_cate2
            output_m23, _ = torch.max(h_m23, dim=-2)#最大池化

        if len(m1_index) == 0:
            output = output_m23
        elif len(m2_index) + len(m3_index) == 0:
            output = output_m1
        else:
            output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
        return output, h_new

