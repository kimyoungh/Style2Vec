"""
    price, turnover, mktcap 데이터를 이용한 멀티팩터 계산 모듈
"""
import datetime
import numpy as np
import pandas as pd
import quantM as qnt

class CalculateMultifactors:
    " multifactor calculator "

    def __init__(self, price, turnover, mktcap):
        " Initialization "

        self.today = datetime.datetime.now()
        self.prev = self.today - datetime.timedelta(days=1)
        self.today = int(self.today.strftime('%Y%m%d'))
        self.prev = int(self.prev.strftime('%Y%m%d'))

        self.price_path = price
        self.turnover_path = turnover
        self.mktcap_path = mktcap

        self.price = None
        self.turnover = None
        self.mktcap = None

        self.codes = None

        self.get_data()

        self.multifactors = None
        self.forders = None
        self.mf_score = None

        self._set_factor_orders()

    def calculate_dataset(self, path=None):
        self.calculate_overall_multifactors()
        self.calculate_multifactor_score()

        self.save(path)

    def save(self, path=None):
        " save data "
        if path == None:
            mf_path = "./dataset/multifactors_" + self.price_path[13:]
            mf_score_path = "./dataset/multifactor_score_" + self.price_path[13:]

        self.multifactors.to_csv(mf_path)
        self.mf_score.to_csv(mf_score_path)

    def calculate_overall_multifactors(self, thres=40):
        " calculate all stocks multifactors "

        for i, code in enumerate(self.codes):
            if i == 0:
                multifactors = self.calculate_multifactors(code, thres=thres)
            else:
                mf = self.calculate_multifactors(code, thres=thres)
                multifactors = multifactors.append(mf)
            print("{}, {}".format(i, code), end='\r')
        print("\n")

        multifactors = pd.DataFrame(multifactors.values, columns=multifactors.columns)

        self.multifactors = multifactors

    def get_data(self):
        "get data"

        price_data = pd.read_csv(self.price_path, index_col=0, header=0)
        turnover_data = pd.read_csv(self.turnover_path, index_col=0, header=0)
        mktcap_data = pd.read_csv(self.mktcap_path, index_col=0, header=0)

        assert((price_data.index == turnover_data.index).sum() ==\
                price_data.shape[0])
        assert((turnover_data.index == mktcap_data.index).sum() ==\
                turnover_data.shape[0])

        self.price = price_data
        self.turnover = turnover_data
        self.mktcap = mktcap_data

        self.codes = self.price.columns.values.tolist()

    def calculate_multifactors(self, code, thres=40):
        " calculate multifactors "
        price = self.price[code]
        turnover = self.turnover[code]
        mktcap = self.mktcap[code]

        returns = np.log(price / price.shift(1)).iloc[1:]

        turnover = turnover.reindex(returns.index)
        mktcap = mktcap.reindex(returns.index)

        multifactors = pd.DataFrame(index=returns.index)
        multifactors['code'] = code
        multifactors['trade_date'] = returns.index.values

        # Price Momentum
        multifactors['pm_5'] = returns.rolling(5).sum()
        multifactors['pm_10'] = returns.rolling(10).sum()
        multifactors['pm_20'] = returns.rolling(20).sum()
        multifactors['pm_40'] = returns.rolling(40).sum()
        multifactors['pm_60'] = returns.rolling(60).sum()
        multifactors['pm_90'] = returns.rolling(90).sum()
        multifactors['pm_120'] = returns.rolling(120).sum()
        multifactors['pm_250'] = returns.rolling(250).sum()

        # Frog in the Pan
        multifactors['fip_5'] = qnt.logfip(returns, p=5)
        multifactors['fip_10'] = qnt.logfip(returns, p=10)
        multifactors['fip_20'] = qnt.logfip(returns, p=20)
        multifactors['fip_40'] = qnt.logfip(returns, p=40)
        multifactors['fip_60'] = qnt.logfip(returns, p=60)
        multifactors['fip_90'] = qnt.logfip(returns, p=90)
        multifactors['fip_120'] = qnt.logfip(returns, p=120)
        multifactors['fip_250'] = qnt.logfip(returns, p=250)

        # Volatility
        multifactors['vol_5'] = returns.rolling(5).std()
        multifactors['vol_10'] = returns.rolling(10).std()
        multifactors['vol_20'] = returns.rolling(20).std()
        multifactors['vol_40'] = returns.rolling(40).std()
        multifactors['vol_60'] = returns.rolling(60).std()
        multifactors['vol_90'] = returns.rolling(90).std()
        multifactors['vol_120'] = returns.rolling(120).std()
        multifactors['vol_250'] = returns.rolling(250).std()

        # Skew
        multifactors['skew_5'] = returns.rolling(5).skew()
        multifactors['skew_10'] = returns.rolling(10).skew()
        multifactors['skew_20'] = returns.rolling(20).skew()
        multifactors['skew_40'] = returns.rolling(40).skew()
        multifactors['skew_60'] = returns.rolling(60).skew()
        multifactors['skew_90'] = returns.rolling(90).skew()
        multifactors['skew_120'] = returns.rolling(120).skew()
        multifactors['skew_250'] = returns.rolling(250).skew()

        # Average Turnover
        turnover_5 = turnover.rolling(5).mean()
        turnover_10 = turnover.rolling(10).mean()
        turnover_20 = turnover.rolling(20).mean()
        turnover_40 = turnover.rolling(40).mean()
        turnover_60 = turnover.rolling(60).mean()
        turnover_90 = turnover.rolling(90).mean()
        turnover_120 = turnover.rolling(120).mean()
        turnover_250 = turnover.rolling(250).mean()

        turnover_5 = turnover_5.apply(lambda x: 1. if x == 0. else x)
        turnover_10 = turnover_10.apply(lambda x: 1. if x == 0. else x)
        turnover_20 = turnover_20.apply(lambda x: 1. if x == 0. else x)
        turnover_40 = turnover_40.apply(lambda x: 1. if x == 0. else x)
        turnover_60 = turnover_60.apply(lambda x: 1. if x == 0. else x)
        turnover_90 = turnover_90.apply(lambda x: 1. if x == 0. else x)
        turnover_120 = turnover_120.apply(lambda x: 1. if x == 0. else x)
        turnover_250 = turnover_250.apply(lambda x: 1. if x == 0. else x)

        multifactors['turnover_5'] = np.log(turnover_5)
        multifactors['turnover_10'] = np.log(turnover_10)
        multifactors['turnover_20'] = np.log(turnover_20)
        multifactors['turnover_40'] = np.log(turnover_40)
        multifactors['turnover_60'] = np.log(turnover_60)
        multifactors['turnover_90'] = np.log(turnover_90)
        multifactors['turnover_120'] = np.log(turnover_120)
        multifactors['turnover_250'] = np.log(turnover_250)

        # Mktcap
        multifactors['log_mktcap'] = np.log(mktcap)

        multifactors = multifactors.iloc[249:]
        null_test = multifactors.isnull().sum(1)
        multifactors = multifactors[null_test != thres]

        return multifactors

    def calculate_multifactor_score(self):
        " calculte multifactor score "
        mf_score = pd.DataFrame(index=self.multifactors.index,
                                columns=self.multifactors.columns)
        mf_score[self.multifactors.columns[:2]] =\
                self.multifactors[['code', 'trade_date']]

        cnt = 0
        for i, date in enumerate(self.multifactors['trade_date'].unique()):
            for j, col in enumerate(self.multifactors.columns[2:]):
                cross_data =\
                        self.multifactors[self.multifactors['trade_date'] ==\
                        date][col]
                normalized = self.normalize_multifactors(cross_data,
                                order=self.forders[col])
                mf_score.loc[normalized.index, col] = normalized
                cnt += 1
                if cnt % 1000 == 0:
                    print(i, j, end='\r')

        # 시가총액 제외 nan 값 0으로 채우기
        mf_score[mf_score.columns[:-1]] =\
                mf_score[mf_score.columns[:-1]].\
                    applymap(lambda x: 0 if pd.isnull(x) else x)
        mf_score = mf_score.dropna()
        self.mf_score = mf_score

    def normalize_multifactors(self, cross_data, order=1):
        """
            minmax scaling

            order:
                1: Desc
                0: Asc
        """
        cmax = max(cross_data)
        cmin = min(cross_data)

        if order == 1:
            normalized = (cross_data - cmin) / (cmax - cmin)
        elif order == 0:
            normalized = (cmax - cross_data) / (cmax - cmin)

        return normalized

    def _set_factor_orders(self):
        " set factor orders "
        self.forders = {}
        self.forders['pm_5'] = 1
        self.forders['pm_10'] = 1
        self.forders['pm_20'] = 1
        self.forders['pm_40'] = 1
        self.forders['pm_60'] = 1
        self.forders['pm_90'] = 1
        self.forders['pm_120'] = 1
        self.forders['pm_250'] = 1

        self.forders['fip_5'] = 1
        self.forders['fip_10'] = 1
        self.forders['fip_20'] = 1
        self.forders['fip_40'] = 1
        self.forders['fip_60'] = 1
        self.forders['fip_90'] = 1
        self.forders['fip_120'] = 1
        self.forders['fip_250'] = 1

        self.forders['vol_5'] = 0
        self.forders['vol_10'] = 0
        self.forders['vol_20'] = 0
        self.forders['vol_40'] = 0
        self.forders['vol_60'] = 0
        self.forders['vol_90'] = 0
        self.forders['vol_120'] = 0
        self.forders['vol_250'] = 0

        self.forders['skew_5'] = 0
        self.forders['skew_10'] = 0
        self.forders['skew_20'] = 0
        self.forders['skew_40'] = 0
        self.forders['skew_60'] = 0
        self.forders['skew_90'] = 0
        self.forders['skew_120'] = 0
        self.forders['skew_250'] = 0

        self.forders['turnover_5'] = 1
        self.forders['turnover_10'] = 1
        self.forders['turnover_20'] = 1
        self.forders['turnover_40'] = 1
        self.forders['turnover_60'] = 1
        self.forders['turnover_90'] = 1
        self.forders['turnover_120'] = 1
        self.forders['turnover_250'] = 1

        self.forders['log_mktcap'] = 1

    def calculate_sharpe_ratio(self, price, period=250, annualize=250):
        """
            Share ratio를 계산하는 함수
            Args:
                price: price series dataframe
                period: 계산 윈도우
        """
        returns = price.pct_change().iloc[1:]

        sharpe = returns.rolling(period).mean() * annualize / \
                (returns.rolling(period).std() * np.sqrt(annualize))

        return sharpe
