import numpy as np
import torch


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=False):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        #print('this is Fm x',x.shape)
        square_of_sum = torch.sum(x, dim=1) ** 2
        #print('square_of_sum',square_of_sum.shape)
        sum_of_square = torch.sum(x ** 2, dim=1)
        #print('sum_of_square',sum_of_square.shape)
        ix = square_of_sum - sum_of_square

        # if self.reduce_sum:
        #     ix = torch.sum(ix, dim=1, keepdim=True)
        #print("this is ix.shape", ix.shape)
        return  ix
class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """


        x1 = self.linear(x) + self.fm(self.embedding(x))
        #print('this is x1_shape',x1.shape)
        #return torch.sigmoid(x.squeeze(1))
        return x1



# import torch
# import numpy as np
# import torch.nn as nn
#
#
# class Feature_Embedding(nn.Module):
#     def __init__(self, field_dims, emb_size):
#         """
#         :param field_dims: 特征数量列表，其和为总特征数量
#         :param emb_size: embedding的维度
#         """
#         super(Feature_Embedding, self).__init__()
#         # embedding层
#         self.emb = nn.Embedding(sum(field_dims), emb_size)
#         # 模型初始化
#         nn.init.xavier_uniform_(self.emb.weight.data)
#         # 偏置项
#         self.offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
#
#     def forward(self, x):
#         # self.offset中存储的是每一列特征计数的开始值
#         # x + x.new_tensor(self.offset)：x中的每一列是分别进行顺序编码+起始值后就可以在self.emb中找到真正的索引
#         x = x + x.new_tensor(self.offset)
#         return self.emb(x)
#
#
# class FactorizationMachine(nn.Module):
#     def __init__(self, field_dims, embed_dim=4):
#         super(FactorizationMachine, self).__init__()
#         self.embed1 = Feature_Embedding(field_dims, 1)
#         self.embed2 = Feature_Embedding(field_dims, embed_dim)
#         self.bias = nn.Parameter(torch.zeros((1,)))
#
#     def forward(self, x):
#         print("this is fm,x",x)
#         # x shape: (batch_size, num_fields)
#         # embed(x) shape: (batch_size, num_fields, embed_dim)
#         # 二阶第一项：和的平方
#         square_sum = self.embed2(x).sum(dim=1).pow(2)
#         print(self.embed2(x).shape)
#         # 二阶第二项：平方的和
#         sum_square = self.embed2(x).pow(2).sum(dim=1)
#         print(sum_square)
#         # 二阶部分
#         second_term = (square_sum - sum_square).sum(dim=1) / 2
#         output = self.embed1(x).squeeze(-1).sum(dim=1) + self.bias + second_term
#         #output = torch.sigmoid(output).unsqueeze(-1)
#         return output
