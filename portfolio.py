"""
    주식 임베딩을 이용한 포트폴리오 모델

    @author: Younghyun Kim
    Created on 2020.05.10
"""
import math
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

class AlphaPort(nn.Module):
    """
        AlphaPort
    """
    def __init__(self, n_styles=16, nheads_ht=4, nlayers_ht=6,
                 sn_indim=320, sn_outdim=320,
                 d_model=320, nheads_cst=8, nlayers_cst=6,
                 dropout=0.2, max_len=5000):
        " Initialization "
        super(AlphaPort, self).__init__()
        self.n_styles = n_styles
        self.nheads_ht = nheads_ht
        self.nlayers_ht = nlayers_ht
        self.sn_indim = sn_indim
        self.sn_outdim = sn_outdim
        self.d_model = d_model
        self.nheads_cst = nheads_cst
        self.nlayers_cst = nlayers_cst
        self.dropout = dropout
        self.max_len = max_len

        self.ht_net = HistoricalTransformer(n_styles, nheads_ht, nlayers_ht)
        self.sum_net = SummaryNet(sn_indim, sn_outdim)
        self.cst_net = CrossSectionalTransformer(d_model, nheads_cst,
                                                 nlayers_cst)
        self.scores_net = nn.Linear(d_model, 1)

        self.pe = PositionalEncoding(n_styles, dropout, max_len=5000)

        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, style_scores):
        ss_pos = self.pe(style_scores)
        hist_stocks = self.ht_net(ss_pos)
        stocks_summary = self.sum_net(hist_stocks)

        stocks_summary = stocks_summary.transpose(0, 1)

        cross_stocks = self.cst_net(stocks_summary).squeeze(0)

        weighting_scores = self.scores_net(cross_stocks)
        weighting_scores = self.relu(weighting_scores)

        weights = self.softmax(weighting_scores).view(-1)

        return weights

    def get_historical_stock_embeddings(self, stock_embeddings):
        """
            stock embedding sequence를 입력 받아
            주식 시계열 임베딩을 반환
        """
        self.eval()
        hist_stocks = self.ht_net(stock_embeddings)

        return hist_stocks

class HistoricalTransformer(nn.Module):
    """
        Historical Transformer
    """
    def __init__(self, n_styles=16, nheads=4, num_layers=6):
        super(HistoricalTransformer, self).__init__()
        self.n_styles = n_styles
        self.nheads = nheads
        self.num_layers = num_layers

        self.encoder_layer = nn.TransformerEncoderLayer(n_styles, nheads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers)

    def forward(self, x):
        hist_stocks = self.transformer_encoder(x)

        return hist_stocks

class SummaryNet(nn.Module):
    """
        Summarization network for HistoricalTransformer
    """
    def __init__(self, input_dim=320, out_dim=320):
        super(SummaryNet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.sum_net = nn.Linear(input_dim, out_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        out = self.sum_net(x)
        out = self.relu(out)

        return out

class CrossSectionalTransformer(nn.Module):
    """
        Cross Sectional Transformer
    """
    def __init__(self, d_model=4000, nheads=8, num_layers=6):
        super(CrossSectionalTransformer, self).__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.num_layers = num_layers

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nheads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers)

    def forward(self, x):
        cross_stocks = self.transformer_encoder(x)

        return cross_stocks

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
