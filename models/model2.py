import math
import pickle
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.utils import DotProductAttention, softAttention,timeattention,logsim,MultiHeadAttention,PositionalEncoding,DotProductAttention2
import torch
import torch.nn as nn




class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

class Classifier(nn.Module):#分类器
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        # pickle.dump(output,open('patient.pkl','wb'))
        # sys.exit()
        if self.activation is not None:
            output = self.activation(output)

        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation,cate_adj,cate_dict,parent_dict
                 ):
        super().__init__()
        self.hidden_size=hidden_size
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size,cate_adj)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size,cate_dict,parent_dict)
        self.us_emb_gender = nn.Embedding(2, hidden_size // 4)
        self.us_emb_Age = nn.Embedding(9, hidden_size // 4)
        self.us_emb_cluster=nn.Embedding(20,hidden_size//8)
        self.position_embedding=PositionalEncoding(150,50)
        #self.attention = DotProductAttention(hidden_size, 32)
        self.attention2 =DotProductAttention2(hidden_size, 32)
        #self.attention=softAttention(hidden_size)
        self.cat_size= hidden_size//4+hidden_size//4+hidden_size+300+hidden_size//8
        #self.cat_size= hidden_size
       # self.cat_size= hidden_size+300
        #self.cat_size= hidden_size
        self.classifier = Classifier(self.cat_size, output_size, dropout_rate, activation)
        self.n1 = torch.nn.BatchNorm1d(self.cat_size)

    def reset_parameters(self,size):
        stdv = 1.0 / math.sqrt(size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, code_x, divided, neighbors, lens,user,cate,text_features,admission_times):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings,cate_embeddings = embeddings
        output = []
        for code_x_i, divided_i, neighbor_i, len_i,user_i ,cate_i,i_text_features ,i_admission_times in zip(code_x, divided, neighbors, lens,user,cate,text_features,admission_times):
            no_embeddings_i_prev = None
            output_i = []
            output_text_i=[]

            h_t = None

            user_emb1 = self.us_emb_gender(user_i[0].long())
            user_emb2 = self.us_emb_Age(user_i[1].long())
            user_emb3=self.us_emb_cluster(user_i[2].long())
            #user_sum = torch.mul(user_emb1, user_emb2)
            for t, (c_it, d_it, n_it, len_it,cate_it,text_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i),cate_i,i_text_features)):

                co_embeddings, no_embeddings ,cao_embeddings= self.graph_layer(c_it, n_it, c_embeddings, n_embeddings,cate_embeddings,cate_it)
                output_it, h_t= self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, cao_embeddings,h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
                output_text_i.append(text_it.to(torch.float32))
            vstack_output = torch.vstack(output_i)

           # vstack_output=self.position_embedding(vstack_output.unsqueeze(0)).squeeze(dim=0)
            output_i=self.attention2(vstack_output)

            #output_i=self.attention(vstack_output)
            #output_i=torch.sum(vstack_output,dim=-2)

            output_i=torch.cat([user_emb1,user_emb2,user_emb3,output_i,output_text_i[-1]])
            #output_i=torch.cat([output_i,output_text_i[-1]])
            #output_i=torch.cat([output_i,output_text_i[-1]])
            #output_i=torch.cat([user_emb3,output_i,output_text_i[-1]])
            output.append(output_i)
        pickle.dump(self.transition_layer.code_tsne,open('nocategorytree.pkl','wb'))
        output = torch.vstack(output)
        output=self.n1(output)
        output = self.classifier(output)
        return output
