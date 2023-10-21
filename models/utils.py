import math

import torch
from torch import nn
import torch.nn.functional as F

class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output

class MultiHeadAttention(nn.Module):
    #graph_size, graph_size, t_output_size, t_attention_size
    def __init__(self,query_size,key_size,value_size,attention_size):
        super().__init__()
        self.attention_size=attention_size
        self.num_heads=8
        self.value_size=value_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)
        self._norm_fact=1/math.sqrt(attention_size)
    def forward(self,q,k,v):
        #print(v.shape)
        n,dim_in=q.shape
       # print(self.value_size)#150
       # print(self.attention_size)
        nh=self.num_heads#8
        dk=self.attention_size//nh#16,
        dv=self.value_size//nh#160//head_num=20
        query = self.dense_q(q).reshape(n,nh,dk).transpose(0,1)#
       # print(query.shape)#[3,16,2]
        key = self.dense_k(k).reshape(n,nh,dk).transpose(0,1)
        value = self.dense_v(v).reshape(n,nh,dv).transpose(0,1)
       # print(value.shape)#([3, 75, 2])

        dist=torch.matmul(query,key.transpose(1,2))*self._norm_fact#key.transpose=[3,2,16]
       # print(dist.shape)#[3,16,16]
        dist=torch.softmax(dist,dim=-1)
       # print(dist.shape)#[3,16,16]
        att=torch.matmul(dist,value)
        att=att.transpose(0,1).reshape(n,self.value_size)
        return att


import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x shape: (seq_len, d_model)
        x = x + self.pe[:x.size(0), :]
        # output shape: (seq_len, d_model)
        return x

class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding2, self).__init__()

        self.d_model = d_model
    def forward(self,len, x):

        pe = torch.zeros(len, self.d_model)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        # x shape: (seq_len, d_model)
        x = x + self.pe[:x.size(0), :].to('cuda')
        # output shape: (seq_len, d_model)
        return x



class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output
class DotProductAttention2(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()

        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(value_size, 1)))
       # self.dense = nn.Linear(value_size, 1)


    def forward(self, x):
        x_split = torch.split(x, x.shape[0], dim=0)
        # x_split_last=self.dense(x_split[-1][-1])
        x_split_last = x_split[-1][-1]

        v_n_repeat = tuple(x_split_last.repeat(x.shape[0], 1))
        v_n_repeat = torch.vstack(v_n_repeat)
        # print(x_split_last.shape)
        # print('v_n',v_n_repeat.shape)
        # print(x_split_last)
        # v_n_repeat=tuple(x_split[-1][-1].view(1,-1).repeat(x.shape[0],1))
        new_embeddings = torch.sigmoid(torch.mul(v_n_repeat, x))
        # print(v_n_repeat.shape)
        # dense_repeat=tuple(self.dense.repeat(x.shape[0],1))
        # dense_repeat=torch.vstack(dense_repeat)
        # new_embeddings=v_n_repeat*self.dense
        # new_embeddings=torch.dot(v_n_repeat,self.dense)
        # print('this is 1',new_embeddings.shape)
        # new_embeddings=torch.mul(x,new_embeddings)
        vu = torch.matmul(new_embeddings, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)

        # t = self.dense(new_embeddings)
        # vu = torch.matmul(t, self.context).squeeze()
        # score = torch.softmax(vu, dim=-1)

        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        # print(output.shape)
        return output
class longandshort(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.q=nn.Linear(hidden_size,1)
        self.W_2=nn.Linear(hidden_size,hidden_size)
        self.W_1=nn.Linear(hidden_size,hidden_size)
        self.W_3=nn.Linear(2*hidden_size,hidden_size)
    def forward(self,x):

        x_split=torch.split(x,x.shape[0],dim=0)
        v_n_repeat=tuple(nodes[-1].view(1,-1).repeat(nodes.shape[0],1) for nodes in x_split)
        temp=self.W_2(torch.cat(v_n_repeat,dim=0))
        #print('temp.shape',temp.shape)
        alpha=self.q(torch.sigmoid(temp+self.W_1(x)))
        #print('alpha.shape',alpha.shape)
        newembedding=alpha*x
        newembedding=torch.sum(newembedding,dim=-2)
        # newembedding_split=torch.split(x,x.shape[0])
        # s_g=tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in newembedding_split)
        # v_n=tuple(nodes[-1].view(1,-1) for nodes in x_split)
        # s_h=self.W_3(torch.cat((torch.cat(v_n,dim=0),torch.cat(s_g,dim=0)),dim=1))
        # newembedding=torch.sum(s_h,dim=-2)
        return newembedding
class timeattention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.timeliner=nn.Linear(1,hidden_size)
        self.time=nn.Linear(1,hidden_size)
        self.q=nn.Linear(hidden_size,1)
        self.w1=nn.Linear(hidden_size,hidden_size)

    def forward(self,time,embeddings,embeddings2):
        #print(time)
        new_embeddings=(self.timeliner(time)+1)*embeddings
        new_embeddings2=(self.time(time)+1)+embeddings2
        finallembeddings=new_embeddings2+new_embeddings
        alpha=self.q(torch.sigmoid(self.w1(finallembeddings)))
        #alpha=self.q(self.w1(finallembeddings))

        new_embeddings=alpha*embeddings

        return new_embeddings
class softAttention2(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.timeliner=nn.Linear(1,hidden_size)
        self.q=nn.Linear(hidden_size,1)

        self.W_1=nn.Linear(hidden_size,hidden_size)
    def forward(self,time,x):
        new_embeddomgs=(self.timeliner(time)+1)*x
        alpha=self.q(torch.sigmoid(self.W_1(x)))
        newembedding=alpha*x
        return newembedding
class softAttention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()

        self.q=nn.Linear(hidden_size,1)
        self.W_1=nn.Linear(hidden_size,hidden_size)

    def forward(self,x):
        alpha=self.q(torch.sigmoid(self.W_1(x)))

        newembedding=alpha*x
        output = torch.sum(newembedding, dim=-2)
        return output
class logsim(nn.Module):
    def __init__(self,user_catsize,hidden_size,note_size):
        super().__init__()
        self.w1=nn.Linear(user_catsize,1)
        self.w2=nn.Linear(hidden_size,1)
        self.w3=nn.Linear(note_size,1)
    def forward(self,user_emb,output_emb,note_emb):

        Gscore=torch.sigmoid(self.w1(user_emb)+self.w2(output_emb)+self.w3(note_emb))
        return Gscore


class timeattention2(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)

        return output

class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()

    def forward(self, H, ht):
        _, channels, _, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size)
        w = self.linear1(new_ht)  # batch_size, 1, filter_num
        conv_vecs = self.conv(H)

        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.relu(conv_vecs)

        # score function
        w = w.expand(-1, self.feat_size, self.filter_num)
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1).expand(-1, self.feat_size, self.filter_num)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)

        concat = torch.cat([ht, v], dim=1)
        new_ht = self.linear2(concat)
        return

class gateAttention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.q=nn.Linear(hidden_size,1)
        self.W_1=nn.Linear(hidden_size,hidden_size)
    def forward(self,x):
        alpha = self.q(torch.sigmoid(self.W_1(x)))
        alpha=alpha.repeat(1,1,2)
        print(alpha)
        alpha[:,:,0]=1-alpha[:,:,0]
        print(alpha)
        g_hat=F.gumbel_softmax(alpha,1)
        print(g_hat)
        g_t=g_hat[:,:1]
        print(g_t)
        breakpoint()
        return g_t

class Dice(nn.Module):
    def __init__(self,
                 emb_size,
                 dim=2,
                 epsilon=1e-8,
                 device='cpu'
                 ):
        super().__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = out.transpose(1, 2)
        return out