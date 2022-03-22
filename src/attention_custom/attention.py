# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 12:44
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : attention.py
# @Software: PyCharm

import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __int__(self, embedding_size, seq_len):
        """
        :param embedding_size: embedding向量的维度, 例如 : 我 => [128] 的向量
        :param seq_len: 单个句子的长度, 单个句子对应的 embedding => [seq_len, embedding_size]
        :return:
        """
        super(SelfAttention, self).__int__()

        self.q = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.k = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.v = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.o = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.embedding_size = embedding_size
        self.seq_len = seq_len

    def forward(self, embedded):
        # 计算 q, k, v
        q = self.q(embedded)
        k = self.k(embedded)
        v = self.v(embedded)

        # 计算注意力权重
        a = torch.matmul(q, k) / torch.sqrt(self.embedding_size)

        # softmax
        a = torch.softmax(a)

        # a @ v
        b = torch.matmul(a, v)
        out = self.o(b)
        return out


if __name__ == '__main__':
    # ====================================
    # input : [seq_len, batch_size, input_size]
    # ====================================
    input_data = torch.randn(6, 256, 128)
    attention = SelfAttention(6, 128)
    out = attention(input_data)
    print(out)
