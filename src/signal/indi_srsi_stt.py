import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt
from functools import reduce
from index import *
from random import shuffle


def srsi_ind(data, N=7):
    VA = data.apply(lambda x: (x.close - x.open) / (x.high - x.low) if x.high != x.low else 0.0, axis=1)
    SRSI = qa.MA(VA, N)
    return pd.DataFrame({'SRSI': SRSI})


def stt_ind(data, N=21, M=42):
    STT = qa.EMA(qa.LINEARREG_SLOPE(data.close, N) * N + data.close, M)
    EMA2 = qa.EMA((data.close * 2 + data.high + data.low) / 4, 2)
    return pd.DataFrame({'STT': STT / EMA2})


if __name__ == '__main__':
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    today_str = today.strftime('%Y-%m-%d')

    start_date = today - dt.timedelta(days=180)
    start_str = start_date.strftime('%Y-%m-%d')

    stocks = qa.QA_fetch_stock_list_adv()
    stock_list = stocks.code.tolist()

    results = []
    strong_stocks = []
    # print('%s: handling %s' % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stock))
    # candles = qa.QA_fetch_stock_day_adv(stock_list, start=start_str, end=today_str).to_qfq()
    for stock in stock_list:
        print('%s: handling %s' % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stock))
        try:
            candles = qa.QA_fetch_stock_day_adv(stock, start=start_str, end=today_str).to_qfq()
            if candles.data.shape[0] <= 50:
                continue
            current = qa.QAFetch.QATdx.QA_fetch_get_stock_latest(stock)
        except:
            print('data error during {}'.format(stock))
            continue
        ptoday = pd.concat([candles.data.reset_index(), current.reset_index(drop=True).rename(columns={'vol': 'volume'})], axis=0).set_index(['date', 'code'])

        # data = candles.data
        # data['c1'] = (data['close'] + data['high'] + data['low']) / 3.
        # data['c2'] = (data['close']*3 + data['high'] + data['low'] + data['open']) / 6.
        # data['c3'] = data['amount'] / data['volume'] / 100.
        # data['cho'] = qa.SUM(data['volume']*(2*data['close']-data['high']-data['low'])/(data['high'] + data['low']), 100)
        # data['zero'] = 0

        srsi = srsi_ind(ptoday)
        stt = stt_ind(ptoday)

        if (srsi.SRSI[-1] > 0 and srsi.SRSI[-2] < 0) and ((stt.STT[-1] > 0 and stt.STT[-2] < 0) or (stt.STT[-2] >0 and stt.STT[-3] < 0)):
            strong_stocks.append(stock)

        results.append(pd.concat([ptoday, srsi, stt], axis=1))

    result = pd.concat(results, axis=0)
    print(strong_stocks)

    result.to_csv('current.csv')
