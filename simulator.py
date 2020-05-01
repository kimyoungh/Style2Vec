"""
    Backtesting Simulator based on embeddings of TIMES

    @author: Younghyun Kim
    @Edited: 2020.04.14.
"""
import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

import torch
import torch.nn as nn

import quantM as qnt

class Simulator:
    """
        Backtesting Simulator for TIMES
    """
    def __init__(self, codes=None, start_date=None, end_date=None,
                 filepath='./dataset/'):
        self.filepath = filepath

        self.start_date = start_date
        self.end_date = end_date
        self.codes = codes

    def calculate_sequence_with_rebalancing(self, returns, weights,
                                            buying_fee=0.0, selling_fee=0.0):
        """
            Backtest에서 수수료 및 리배런싱 적용한
            바스켓 변화과정 계산하여 배열 반환
            returns: 투자 대상 종목들의 로그수익률 데이터프레임
                     weights보다 넓은 범위의 종목들을 포함해야 함
            weights: dataframe. 리밸런싱 투자비중 데이터,
                     리밸런싱 시점에 대한 데이터만 포함
            buying_fee: 매수 수수료
            selling_fee: 매도 수수료(세금 포함)
            weights 시점이 returns보다 한 시점 빠름
            예: 2011년 1월 30일 종가 기준으로 투자비중 설정한 월간 수익률
            기준 포트폴리오에서 weights가 returns에 실제로 곱해지는
            날짜는 2011년 2월 28일임

            * 전 종목에 100% 투자했다고 가정

            Returns:
                port: 포트폴리오 시점 별 절대수익률 데이터프레임
                cum_port_returns: 포트폴리오 누적절대수익률 시계열 데이터프레임
                weights: 시점 별 시작 시점 기준 포트폴리오 종목 비중 시계열
                         (일자 데이터일 경우, 전일 종가 기준 포트폴리오 종목 비중)
        """

        if buying_fee >= 0.1:
            buying_fee /= 100.
        if selling_fee >= 0.1:
            selling_fee /= 100.

        # weights를 returrns크기에 맞추기
        returns_index = returns.index
        weights_index = weights.index
        x_weights_index = returns_index.append(weights_index).unique()
        x_weights_index = x_weights_index.sort_values()

        weights_seq = np.zeros((x_weights_index.shape[0], returns.shape[1]))
        weights_seq = pd.DataFrame(weights_seq, index=x_weights_index,
                                   columns=returns.columns)
        weights_seq.loc[weights.index, weights.columns] = weights

        weights = weights_seq.shift(1).iloc[1:]

        if weights.index[0] is not returns.index[0]:
            raise \
            ValueError("Initial dates of weights and returns are not matching!")

        weights = weights.applymap(lambda x: 0. if pd.isnull(x) else x)
        returns = returns.applymap(lambda x: 0. if pd.isnull(x) else x)
        returns = returns.reindex(index=weights.index)

        # 리밸런싱 시점 위치 정보 가져오기
        rebalance_t = []
        for i, _ in enumerate(weights.index):
            if weights.iloc[i].sum() > 0.:
                rebalance_t.append(i)

        weights_v = weights.values
        returns_v = returns.values

        weights_diff = np.zeros_like(weights_v)
        fee_ind = np.zeros_like(weights_v)  # 각 종목 수수료 모음

        for i, weight in enumerate(weights_v):
            if i == 0:
                for j in range(weights_v.shape[1]):
                    if weights_v[i, j] > 0:
                        weights_diff[i, j] = weight[j]
                        fee_ind[i, j] = buying_fee
            elif (i > 0) and (i not in rebalance_t):
                weights_v[i] = weights_v[i - 1] * np.exp(returns_v[i - 1])

            elif (i > 0) and (i in rebalance_t):
                for j in range(weights_v.shape[1]):
                    if weights_v[i, j] > weights_v[i - 1, j]:  # Buy more
                        weights_diff[i, j] =\
                                weights_v[i, j] - weights_v[i - 1, j]
                        fee_ind[i, j] = buying_fee
                    elif weights_v[i, j] < weights_v[i - 1, j]:  # Sell more
                        weights_diff[i, j] =\
                                weights_v[i - 1, j] - weights_v[i, j]
                        fee_ind[i, j] = selling_fee
                    else:
                        weights_diff[i, j] =\
                                weights_v[i, j] - weights_v[i - 1, j]
                        fee_ind[i, j] = 0.
            weights_v[i] /= weights_v[i].sum()
        fees = (weights_diff * fee_ind).sum(1)

        # 수수료가 적용된 수익률과 투자비중을 곱한 최종 수익률 결과
        port_returns = (np.exp(returns_v) - 1) * weights_v
        port_returns = pd.DataFrame(port_returns, index=weights.index,
                                    columns=weights.columns)

        weights = pd.DataFrame(weights_v, index=port_returns.index,
                               columns=port_returns.columns)

        # 포트폴리오 수익률 시계열
        port = port_returns.sum(1) - fees

        # 누적 수익률 시계열
        cum_port_returns = (1 + port).cumprod() - 1.

        return port, cum_port_returns, weights

    def calculate_style_score_ew_portfolio(self, style_scores,
                                           codes, topK=10):
        """
            Style Score 기준 상위 topK 종목을 뽑아
            t 시점 동일가중 포트폴리오를 만드는 함수

            Args:
                style_scores: np.ndarray. 종목 별 style_score 배열
                codes: np.ndarray. 종목 별 코드(style_scores와 종목 순서 일치)
                topK: 상위 종목 수 (default: 10)

            Return:
                invested_stocks: 투자된 종목 목록 np.ndarray
                weights: 선정 종목 투자 비중 np.ndarray
        """
        assert style_scores.shape[0] == codes.shape[0]

        rankings = style_scores.argsort()[::-1]
        invested_stocks = codes[rankings[:topK]]

        weights = np.array([1. / topK] * topK)

        return invested_stocks, weights

    def calculate_style_score_ls_ew_portfolio(self, style_scores,
                                              codes, topK=10, bottomK=10):
        """
            Style score 기준 topK, bottomK 종목을 뽑아
            t 시점 동일가중 포트폴리오를 만드는 함수

            Args:
                style_scores: np.ndarray. 종목 별 style_score 배열
                codes: np.ndarray. 종목 별 코드(style_scores와 종목 순서 일치)
                topK: 상위 종목 수 (default: 10)

            Return:
                long_stocks, short_stocks, weights_long, weights_short,
        """
        assert style_scores.shape[0] == codes.shape[0]

        rankings = style_scores.argsort()[::-1]
        long_stocks = codes[rankings[:topK]]
        short_stocks = codes[rankings[-bottomK:]]

        weights_long = np.array([1. / topK] * topK)
        weights_short = np.array([1. / bottomK] * bottomK)

        return long_stocks, short_stocks, weights_long, weights_short

    def calculate_style_equal_weight_portfolio(self, style_embedding,
                                               stock_embeddings,
                                               codes, topK=10):
        """
            Style Embedding과 가까운 topK 종목을 뽑아
            t 시점 동일가중 포트폴리오를 만드는 함수

            Args:
                style_embedding: style_embedding(1 X stock_dim)
                stock_embeddings: 전체 종목 stock embedding의
                                  matrix(N X stock_dim)
                codes: 전체 종목 코드 np.ndarray(1 X N)
                       -> stock_embeddings row와 codes의 column이
                          같은 종목 순서로 매칭되어야 함
                topK: 유사도 상위 종목 수 (default: 10)

            Return:
                invested_stocks: 투자된 종목 목록 np.ndarray
                weights: 선정 종목 투자 비중 np.ndarray
                sim_desc: 선정 종목들과 스타일 임베딩 간의 cosine similarity
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        sim = cos(style_embedding, stock_embeddings)

        sim_desc, sim_ranks = sim.sort(descending=True)
        sim_desc = sim_desc.detach().numpy()
        sim_ranks = sim_ranks().detach().numpy()

        weights = np.array([1. / topK] * topK)
        invested_stocks = codes[sim_ranks[:topK]]

        return invested_stocks, weights, sim_desc[sim_ranks[:topK]]

    def calculate_style_longshort_ew_portfolio(self, style_embedding,
                                               stock_embeddings,
                                               codes, topK=10, bottomK=10):
        """
            Style embeddings와 가까운 topK, bottomK 종목을 뽑아
            t 시점 동일가중 포트폴리오를 만드는 함수

            Args:
                style_embeddiing: style_embedding(1 X stock_dim)
                stock_embeddings: 전체 종목 stock embedding의
                                  matrix(N X stock_dim)
                codes: 전체 종목 코드 np.ndarray(1 X N)
                       -> stock_embeddings row와 codes의 column이
                          같은 종목 순서로 매칭되어야 함
                topK: 유사도 상위 종목 수(default: 10)
                bottomK: 유사도 하위 종목 수(default: 10)

            Return:
                long_stocks, short_stocks, weights_long, weights_short,
                sim_desc_long, sim_desc_short
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        sim = cos(style_embedding, stock_embeddings)

        sim_desc, sim_ranks = sim.sort(descending=True)
        sim_desc = sim_desc.detach().numpy()
        sim_ranks = sim_ranks.detach().numpy()

        weights_long = np.array([1. / topK] * topK)
        weights_short = np.array([1. / bottomK] * bottomK)

        long_stocks = codes[sim_ranks[:topK]]
        short_stocks = codes[sim_ranks[-bottomK:]]

        return long_stocks, short_stocks, weights_long, weights_short, \
                sim_desc[sim_ranks[:topK]], sim_desc[sim_ranks[-bottomK:]]
