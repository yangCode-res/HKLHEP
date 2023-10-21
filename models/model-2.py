import math

import torch
from torch import nn

from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.utils import DotProductAttention, softAttention, gateAttention, Dice, DotProductAttention2


class Classifier(nn.Module):#分类器
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        #print(x.shape)
        output = self.dropout(x)

        output = self.linear(output)


        #print(output)
        if self.activation is not None:
            output = self.activation(output)

        #print(output)
        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation,cate_adj,cate_dict
                 ):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size,cate_adj)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size,cate_dict)
        self.attention = DotProductAttention(hidden_size, 32)
        self.us_emb_gender=nn.Embedding(2,hidden_size//4)
        self.us_emb_Age=nn.Embedding(13,hidden_size//4)
        self.us_emb_cluster=nn.Embedding(10,hidden_size//8)
        self.cat_size=hidden_size+hidden_size//4+hidden_size//4+hidden_size//8+300
        self.softattention=softAttention(hidden_size)
        self.attention2=DotProductAttention2(300,300)
        self.classifier = Classifier(self.cat_size, output_size, dropout_rate, activation)
        self.n1=torch.nn.BatchNorm1d(self.cat_size)

    def reset_parameters(self,size):
        stdv = 1.0 / math.sqrt(size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, code_x, divided, neighbors, lens,user,cate,text_features,event_types):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings,cate_embeddings = embeddings
        output = []
        for code_x_i, divided_i, neighbor_i, len_i,user_i ,cate_i,i_text_features,i_event_types in zip(code_x, divided, neighbors, lens,user,cate,text_features,event_types):
            no_embeddings_i_prev = None
            output_i = []
            output_text_i=[]
            h_t = None
            user_emb1 = self.us_emb_gender(user_i[0].long())
            user_emb2 = self.us_emb_Age(user_i[1].long())
            user_emb3=self.us_emb_cluster(user_i[2].long())
            #print(i_text_features.shape)[32,300]
            for t, (c_it, d_it, n_it, len_it,cate_it,text_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i),cate_i,i_text_features)):

                co_embeddings, no_embeddings ,cao_embeddings= self.graph_layer(c_it, n_it, c_embeddings, n_embeddings,cate_embeddings,cate_it)
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, cao_embeddings,h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
                output_text_i.append(text_it.to(torch.float32))

            vstack_output_text=torch.vstack(output_text_i)
            vstack_output = torch.vstack(output_i)

            vstack_output_text=self.attention2(vstack_output_text).squeeze()



            vstack_output=self.softattention(vstack_output)
            output_i=self.attention(vstack_output)
            output_i=torch.cat([user_emb1,user_emb2,user_emb3,vstack_output_text,output_i])

            output.append(output_i)

        output = torch.vstack(output)
        output=self.n1(output)
        output = self.classifier(output)
        return output
