import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt
from functools import reduce


def ma_lines(data, N):
    return pd.DataFrame({'MA': data.close / qa.MA(data.close, N), 'EMA': data.close / qa.EMA(data.close, N), 'SMA': data.close / qa.SMA(data.close, N)})


def boll_lines(data, N=20):
    boll = qa.MA(data.close, N)
    UB = boll + 2 * qa.STD(data.close, N)
    LB = boll - 2 * qa.STD(data.close, N)
    DICT = {'UB': data.close / UB, 'LB': data.close / LB}
    return pd.DataFrame(DICT)


if __name__ == '__main__':
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    today_str = today.strftime('%Y-%m-%d')

    # divide_date = today - dt.timedelta(days=60)
    # divide_str = divide_date.strftime('%Y-%m-%d')

    stocks = qa.QA_fetch_stock_list_adv()
    stock_list = stocks.code.tolist()
    for stock in stock_list[:10]:
        print('handling %s' % stock)
        try:
            candles = qa.QA_fetch_stock_day_adv(stock, start='2013-01-01', end=today_str).to_qfq()
            if candles.data.shape[0] <= 100:
                continue
        except:
            print('data error during {}'.format(stock))
            continue

        data = candles.data
        # price lines, will be close / line
        ma = [candles.add_func(ma_lines, N=5).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_5', 'EMA': 'ema_5', 'SMA': 'sma_5'}),
              candles.add_func(ma_lines, N=8).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_8', 'EMA': 'ema_8', 'SMA': 'sma_8'}),
              candles.add_func(ma_lines, N=10).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_10', 'EMA': 'ema_10', 'SMA': 'sma_10'}),
              candles.add_func(ma_lines, N=13).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_13', 'EMA': 'ema_13', 'SMA': 'sma_13'}),
              candles.add_func(ma_lines, N=15).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_15', 'EMA': 'ema_15', 'SMA': 'sma_15'}),
              candles.add_func(ma_lines, N=18).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_18', 'EMA': 'ema_18', 'SMA': 'sma_18'}),
              candles.add_func(ma_lines, N=20).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_20', 'EMA': 'ema_20', 'SMA': 'sma_20'}),
              candles.add_func(ma_lines, N=21).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_21', 'EMA': 'ema_21', 'SMA': 'sma_21'}),
              candles.add_func(ma_lines, N=30).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_30', 'EMA': 'ema_30', 'SMA': 'sma_30'}),
              candles.add_func(ma_lines, N=34).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_34', 'EMA': 'ema_34', 'SMA': 'sma_34'}),
              candles.add_func(ma_lines, N=40).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_40', 'EMA': 'ema_40', 'SMA': 'sma_40'}),
              candles.add_func(ma_lines, N=44).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_44', 'EMA': 'ema_44', 'SMA': 'sma_44'}),
              candles.add_func(ma_lines, N=50).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_50', 'EMA': 'ema_50', 'SMA': 'sma_50'}),
              candles.add_func(ma_lines, N=55).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_55', 'EMA': 'ema_55', 'SMA': 'sma_55'}),
              candles.add_func(ma_lines, N=60).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_60', 'EMA': 'ema_60', 'SMA': 'sma_60'}),
              candles.add_func(ma_lines, N=66).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_66', 'EMA': 'ema_66', 'SMA': 'sma_66'}),
              candles.add_func(ma_lines, N=89).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_89', 'EMA': 'ema_89', 'SMA': 'sma_89'}),
              candles.add_func(ma_lines, N=99).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_99', 'EMA': 'ema_99', 'SMA': 'sma_99'}),
              candles.add_func(ma_lines, N=120).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_120', 'EMA': 'ema_120', 'SMA': 'sma_120'}),
              candles.add_func(ma_lines, N=144).loc[:, ['MA', 'EMA', 'SMA']].rename({'MA': 'ma_144', 'EMA': 'ema_144', 'SMA': 'sma_144'})]

        boll = [candles.add_func(boll_lines, N=5).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_5', 'LB': 'boll_lb_5'}),
                candles.add_func(boll_lines, N=9).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_9', 'LB': 'boll_lb_9'}),
                candles.add_func(boll_lines, N=13).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_13', 'LB': 'boll_lb_13'}),
                candles.add_func(boll_lines, N=20).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_20', 'LB': 'boll_lb_20'}),
                candles.add_func(boll_lines, N=25).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_25', 'LB': 'boll_lb_25'}),
                candles.add_func(boll_lines, N=30).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_30', 'LB': 'boll_lb_30'}),
                candles.add_func(boll_lines, N=35).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_35', 'LB': 'boll_lb_35'}),
                candles.add_func(boll_lines, N=40).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_40', 'LB': 'boll_lb_40'}),
                candles.add_func(boll_lines, N=50).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_50', 'LB': 'boll_lb_50'}),
                candles.add_func(boll_lines, N=60).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_60', 'LB': 'boll_lb_60'}),
                candles.add_func(boll_lines, N=75).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_75', 'LB': 'boll_lb_75'}),
                candles.add_func(boll_lines, N=99).loc[:, ['UB', 'LB']].rename({'UB': 'boll_ub_99', 'LB': 'boll_lb_99'})]

        # quantiles
        cci = [candles.add_func(qa.QA_indicator_CCI, N=x).loc[:, 'CCI'].rename(columns={'CCI': 'cci_{}'.format(x)}) for x in range(6, 101, 2)]

        macd = [candles.add_func(qa.QA_indicator_MACD, short=x, long=y, mid=z).loc[:, ['DIF', 'DEA', 'MACD']].rename(columns={'DIF': 'dif_{}_{}_{}'.format(x, y, z),
                                                                                                                              'DEA': 'dea_{}_{}_{}'.format(x, y, z),
                                                                                                                              'MACD': 'macd_{}_{}_{}'.format(x, y, z)})
                for x in range(3, 31, 3) for y in range(4, 61, 2) for z in range(3, 31, 3) if x != y]
        skdj_9_3 = candles.add_func(qa.QA_indicator_SKDJ, N=9, M=3).loc[:, ['SKDJ_K', 'SKDJ_D']].rename(columns={'SKDJ_D': 'skdj_d_9_3', 'SKDJ_K': 'skdj_k_9_3'})


