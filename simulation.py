"""
    Backtesting Simulator based on embeddings of TIMES 

    @author: Younghyun Kim
    @Edited: 2020.04.14.
"""
import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import pyarrow as pa
from pyarrow import parquet as pq

import torch
import torch.nn as nn

from kb_common.kb_config import constants
from kb_common.db_manager import DBExecuteManager

from multifactor_dataloader import MultiFactorDataLoader

class Simulator:
    """
        Backtesting Simulator for TIMES
    """
    def __init__(self, codes=None, start_date=None, end_date=None,
                 ric='005930.KS', filepath='./dataset/'):
        """ Initialization """
        self.filepath = filepath

        # Set Dataloader
        self.dataloader = MultiFactorDataLoader(codes, start_date,
                                                end_date, ric, filepath)

        if start_date is None:
            self.start_date = self.dataloader.start_date
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = self.dataloader.end_date
        else:
            self.end_date = end_date

        self.codes = self.dataloader.codes
        self.price_dates = self.dataloader.price_dates
        self.mf_dates = self.dataloader.mf_dates
        self.dates = self.dataloader.dates
        self.factors = self.dataloader.factors

    def calculate_sequence_with_rebalancing(self, returns, weights,
                                            buying_fee=0.0, selling_fee=0.0):
        """
            Backtest에서 수수료 및 리밸런싱을 적용한
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

        # weights를 returns 크기에 맞추기
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

    def calculate_score_based_ew_portfolio(self, score_data, rics, topK=10):
        """
            Score 기반 topK 종목을 뽑아
            t 시점 동일가중 포트폴리오를 만드는 함수
            Args:
                score_data: np.ndarray. 종목 별 score data(1 X N),
                            선호되는 종목에 높은 스코어 부여되어야 함
                rics: np.ndarray. 전체 종목 코드(1 X N), score_data 위치와 대응되어야 함
                topK: 스코어 상위 종목 수(default: topK=10)

            Returns:
                invested_stocks: np.ndarray. 투자된 종목 목록
                weights: 선정 종목 투자 비중 np.ndarray
        """
        score_rank = score_data.argsort()[::-1]

        weights = np.array([1. / topK] * topK)

        invested_stocks = rics[score_rank[:topK]]

        return invested_stocks, weights

    def calculate_score_based_longshort_ew_portfolio(self, score_data, rics,
                                                     topK=10, bottomK=10):
        """
            Score 기반 topK, bottomK 종목을 뽑아
            t 시점 동일가중 기반 롱숏 포트폴리오를 만드는 함수

            Args:
                score_data: np.ndarray. 종목 별 score data(1 X N),
                            선호되는 종목에 높은 스코어 부여되어야 함
                rics: np.ndarray. 전체 종목 코드(1 X N), score_data 위치와 대응되어야 함
                topK: 스코어 상위 종목 수(default: topK=10)

            Returns:
                long_stocks: 매수 종목 목록 np.ndarray
                short_stocks: 매도 종목 목록 np.ndarray
                weights_long: 매수 종목 투자 비중 np.ndarray
                weights_short: 매도 종목 투자 비중 np.ndarray
        """
        score_rank = score_data.argsort()[::-1]

        weights_long = np.array([1. / topK] * topK)
        weights_short = np.array([1. / bottomK] * bottomK)

        long_stocks = rics[score_rank[:topK]]
        short_stocks = rics[score_rank[-bottomK:]]

        return long_stocks, short_stocks, weights_long, weights_short

    def calculate_style_equal_weight_portfolio(self, style_embedding,
                                               stock_embeddings,
                                               rics, topK=10):
        """
            Style Embedding과 가까운 topK 종목을 뽑아
            t 시점 동일가중 포트폴리오를 만드는 함수

            Args:
                style_embedding: torch.tensor. style_embedding(1 X stock_dim)
                stock_embeddings: torch.tensor. 전체 종목 stock embedding의
                                  matrix(N X stock_dim)
                rics: 전체 종목 코드 np.ndarray(1 X N)
                       -> stock_embeddings row와 rics의 column이
                같은 종목 순서로 매칭되어야 함
                topK: 유사도 상위 종목 수(default: 10)

            Return:
                invested_stocks: 투자된 종목 목록 np.ndarray
                weights: 선정 종목 투자 비중 np.ndarray
                sim_desc: 선정 종목들과 스타일 임베딩 간의 cosine similarity
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        sim = cos(style_embedding, stock_embeddings)

        sim_desc, sim_ranks = sim.sort(descending=True)
        sim_desc = sim_desc.detach().numpy()
        sim_ranks = sim_ranks.detach().numpy()

        weights = np.array([1. / topK] * topK)
        invested_stocks = rics[sim_ranks[:topK]]

        return invested_stocks, weights, sim_desc[sim_ranks[:topK]]

    def calculate_style_longshort_ew_portfolio(self, style_embedding,
                                               stock_embeddings,
                                               rics, topK=10, bottomK=10):
        """
            Style embedding과 가까운 topK, bottomK 종목을 뽑아
            t 시점 동일가중 포트폴리오를 만드는 함수

            Args:
                style_embedding: style_embedding(1 X stock_dim)
                stock_embeddings: 전체 종목 stock embedding의
                                  matrix(N X stock_dim)
                rics: 전체 종목 코드 np.ndarray(1 X N)
                       -> stock_embeddings row와 rics의 column이
                같은 종목 순서로 매칭되어야 함
                topK: 유사도 상위 종목 수(default: 10)
                bottomK: 유사도 하위 종목 수(default: 10)

            Return:
                long_stocks: 매수 종목 목록 np.ndarray
                short_stocks: 매도 종목 목록 np.ndarray
                weights_long: 매수 종목 투자 비중 np.ndarray
                weights_short: 매도 종목 투자 비중 np.ndarray
                sim_desc[sim_ranks[:topK]]: 매수 종목들과 스타일 임베딩 간의
                                            cosine similarity
                sim_desc[sim_ranks[-bottomK:]]: 매도 종목들과 스타일 임베딩 간의
                                                cosine similarity
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        sim = cos(style_embedding, stock_embeddings)

        sim_desc, sim_ranks = sim.sort(descending=True)
        sim_desc = sim_desc.detach().numpy()
        sim_ranks = sim_ranks.detach().numpy()

        weights_long = np.array([1. / topK] * topK)
        weights_short = np.array([-1. / bottomK] * bottomK)

        long_stocks = rics[sim_ranks[:topK]]
        short_stocks = rics[sim_ranks[-bottomK:]]

        return long_stocks, short_stocks, weights_long, weights_short, \
                sim_desc[sim_ranks[:topK]], sim_desc[sim_ranks[-bottomK:]]

    def calculate_average_turnover(self, weights):
        """
            Calculate average turnover rate based on timeseries for portfolio
            weights.

            Args:
                weights: pd.DataFrame. Timeseries for portfolio
                         * 최소 시점단위가 누락되면 안됨(250일 기간이면
                           250일 치가 다 있어야 함)
            Return:
                turnover: Average turnover rate
        """
        weights_diff = abs(weights - weights.shift(1))
        turnover = weights_diff.mean().mean()

        return turnover

    def calculate_traversal_correlation(self, style_score_traversed,
                                        mf_scores, thres=0.4):
        """
            Spearman Correlation Coefficient를 이용하여
            스타일 스코어와 멀티팩터 스코어의 관계 뽑아내기
            (monotone or non-monotone)

            Args:
                style_score_traversed: 단일 스타일 스코어 조절 값
                mf_scores: style_score_traversed 값에 대응하는
                          멀티팩터 스코어 변화 값
                thres: 관계 유무 기준 값(default: 0.4)

            Return:
                corrs: style score와 멀티팩터 순서별 관계
        """
        if style_score_traversed.shape[0] is not mf_scores.shape[0]:
            raise ValueError(
                "The length of style_score_traversed is not equal to that of mf_scores")

        traversed_scores = np.zeros((mf_scores.shape[0],
                                     mf_scores.shape[1] + 1))
        traversed_scores[:, 0] = style_score_traversed
        traversed_scores[:, 1:] = mf_scores

        corr, _ = spearmanr(traversed_scores)
        corr = corr[1:, 0].ravel()

        corrs = np.zeros((corr.shape[0], 3))
        corrs[:, 0] = np.arange(corr.shape[0])
        corrs[:, 1] = corr  # Spearman Correlation Coefficient
        corrs[:, 2] = abs(corr) >= thres  # Threshold

        corrs = corrs[corrs[:, 2] > 0]
        corrs = corrs[corrs[:, 1].argsort()]

        factors = np.array(self.dataloader.factors)

        corrs = pd.DataFrame(corrs[:, 1], columns=['Correlation'],
                             index=factors[corrs[:, 0].astype(int)])

        return corrs[::-1]

    def get_overall_data(self, codes=None, start_date=None, end_date=None):
        """
            Get multifactor score data and sharpe ratio data

            Return:
                mf_index, mf_scores, sratio
        """
        mf_index, mf_scores, sratio = self.dataloader.get_overall_data(codes,
                                                                       start_date,
                                                                       end_date)

        return mf_index, mf_scores, sratio

    def get_multifactor_scores_data(self, codes=None, start_date=None,
                                    end_date=None):
        """
            Get multifactor data

            Return:
                mf_index: list of tuple(ric, date)
                mf_scores: np.ndarray multifactor score data(equal index
                with mf_index)
        """
        mf_index, mf_scores = self.dataloader.get_mf_score_data(codes,
                                                                start_date,
                                                                end_date)

        return mf_index, mf_scores

    def get_multiple_log_returns(self, codes=None,
                                 start_date=None, end_date=None):
        """
            여러 종목의 로그수익률 데이터 가져오기
        """
        price = self.get_multiple_price(codes, start_date, end_date)

        returns = np.log(price / price.shift(1)).iloc[1:]

        return returns

    def get_multiple_price(self, codes=None, start_date=None, end_date=None):
        """
            여러 종목의 주가 데이터 가져오기

            Args:
                codes: 종목 코드 리스트
                start_date: 데이터 시작일
                end_date: 데이터 종료일
            Return:
                price: 주가 데이터프레임
        """
        price = self.dataloader.get_multiple_price(codes, start_date, end_date)
        price = price.astype(np.float)

        return price

    def get_price(self, code, start_date, end_date):
        """
            특정 종목의 주가 데이터 가져오기

            Args:
                code: 종목 코드
                start_date: 데이터 시작일
                end_date: 데이터 종료일

            Return:
                price: 주가 데이터프레임

            # 시계열 중간에 주가 없는 경우 전날 주가 반영
        """
        price = self.dataloader.get_price(code, start_date, end_date)

        return price

    def get_month_dates(self, dates):
        """
            get monthly dates
        """
        if len(dates) < 2:
            raise ValueError("The length has to be longer than 1")
        months = []
        for i, date in enumerate(dates):
            if i < len(dates) - 1:
                if date.month is not dates[i + 1].month:
                    months.append(date)
            else:
                months.append(date)

        return months

    def get_quarter_dates(self, dates):
        """
            get quarter dates
        """
        months = self.get_month_dates(dates)

        quarters = []
        for date in months:
            if date.month == 3 or date.month == 6 \
                    or date.month == 9 or date.month == 12:
                quarters.append(date)

        return quarters
