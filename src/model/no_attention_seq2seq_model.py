# -*- coding: utf-8 -*-
"""
@File : no_attention_lstm_model.py
@Auth : 王天赐
@Time : 2022/3/16/0016 15:47
@IDE  : PyCharm
@Desc : 
"""

import torch
import torch.nn as nn
from datasets.dataset_ import MAX_LENGTH

# 判断当前机器是CPU还是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN():

    def __init__(self, input_size, hidden_size):
        """
        :param input_size:
        :param hidden_size:
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 初始化 Embedding 层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 初始化 GRU层
