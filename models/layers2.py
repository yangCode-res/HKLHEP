import pickle

import numpy
import numpy as np
import torch
import tqdm
from torch import nn

from models.utils import SingleHeadAttentionLayer


class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))#疾病的embedding
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))#邻居的embedding
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))#不相关的embedding
        self.catecory_embedding=nn.Embedding(300,code_size)
    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings,self.catecory_embedding


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size*2, graph_size)
        self.activation = nn.LeakyReLU()
        self.catecory=pickle.load(open("D:\pythonprojets\Chet-master\models\encode_icd9tocatecory.pkl",'rb'))
    def forward(self, code_x, neighbor, c_embeddings, n_embeddings,catecory_embeddings):
        device = torch.device('cuda')

        #print("codex",code_x.shape)
        numpy.set_printoptions(threshold=np.inf)
        center_codes = torch.unsqueeze(code_x, dim=-1)#[1,4880]

        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)#[1,4880]

        center_embeddings = center_codes * c_embeddings#mt*M,eq2的第一项#[4880,48]



        catecory_code_embeddings = np.zeros((4880,28),dtype=float)
        catecory_code_embeddings = torch.Tensor(catecory_code_embeddings).to(device)
        for i,code in enumerate(center_codes):
            if(code==True) :
                catecory_code_embeddings[i]=catecory_embeddings(torch.tensor(self.catecory[i]).to(device))
        catecory_embeddings=center_codes*catecory_code_embeddings
        center_embeddings=torch.cat([center_embeddings,catecory_code_embeddings],-1)
        #print(center_embeddings.shape)



        #print("this is center_embedding",center_embeddings)
        #print("this is center_embeddingsHape",center_embeddings.shape)
        neighbor_embeddings = neighbor_codes * n_embeddings#eq3的第一项[4880,48]
        #print(torch.matmul(self.adj,center_embeddings).shape)
        #print(self.adj.shape)[4880,4880]matmul(4880*48)
        catecory_neighbor_embeddings=np.zeros((4880,28),dtype=float)
        catecory_neighbor_embeddings = torch.Tensor(catecory_neighbor_embeddings).to(device)
        for i,code in enumerate(neighbor_codes):
            if(code==True) :
                catecory_neighbor_embeddings[i]=catecory_embeddings[self.catecory[i]]

        catecory_neighbor_embeddings=neighbor_codes*catecory_neighbor_embeddings
        neighbor_embeddings=torch.cat([neighbor_embeddings,catecory_neighbor_embeddings],-1)
        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)#local context[4880,48]
        cn_embeddings = center_codes * torch.matmul(self.adj, neighbor_embeddings)#diagnosis global context
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)#eq2的第二项
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)#eq3的第三项
        #print(cn_embeddings.shape)
        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings))#eq2本地疾病的embdding
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings))#eq3邻居的embedding
        return co_embeddings, no_embeddings


class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()
        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, hidden_state=None):
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]#m1是persistent diseases
        m1_index = torch.where(m1 > 0)[0]#m2是emerging neighbors
        m2_index = torch.where(m2 > 0)[0]#m3是emerging unrelated diseases
        m3_index = torch.where(m3 > 0)[0]
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
        output_m1 = 0
        output_m23 = 0
        if len(m1_index) > 0:
            m1_embedding = co_embeddings[m1_index]
            h = hidden_state[m1_index] if hidden_state is not None else None
            h_m1 = self.gru(m1_embedding, h)
            h_new[m1_index] = h_m1
            output_m1, _ = torch.max(h_m1, dim=-2)
        if t > 0 and len(m2_index) + len(m3_index) > 0:
            q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
            v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
            h_m23 = self.activation(self.single_head_attention(q, q, v))
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]
            output_m23, _ = torch.max(h_m23, dim=-2)#最大池化
        if len(m1_index) == 0:
            output = output_m23
        elif len(m2_index) + len(m3_index) == 0:
            output = output_m1
        else:
            output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
        return output, h_new

