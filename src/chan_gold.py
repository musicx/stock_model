import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt
from utility import *


if __name__ == '__main__':
    # today = dt.datetime.today()
    today = dt.datetime(2018, 7, 25)
    start_date = today - dt.timedelta(days=365)

    # stocks = ['000528', '002049', '300529', '300607', '600518', '600588', '603877']
    # stocks = ['300638', '600516']
    stocks = ['600997']

    with open('../data/chan_analysis.txt', 'w') as f:
        for stock in stocks:
            can = qa.QA_fetch_stock_day_adv(stock, start=start_date.strftime('%Y-%m-%d'),
                                            end=today.strftime('%Y-%m-%d')).to_qfq()
            raw = can.data.loc[:, ['open', 'close', 'high', 'low']]
            try:
                raw['dd'] = [x.strftime('%Y-%m-%d') for x in can.data.reset_index().date.tolist()]
            except:
                raw['dd'] = [x.strftime('%Y-%m-%d') for x in can.data.date.tolist()]
            klines = [Kline(x[0], x[1], x[2], x[3], x[4]) for x in raw.values]

            f.write('stock code: {}\n'.format(stock))

            merged = merge_klines(klines)
            ends = find_endpoints(merged)

            for point in ends:
                f.write('{}: {} {} end\n'.format(merged[point[0]].date,
                                                 'valid' if point[2] else 'invalid',
                                                 'top' if point[1] else 'bottom'))

            cycles = find_cycle(ends, merged)
            print("found cycles: {}".format(','.join(map(str, cycles))))

            pcycles = find_pair_cycle(cycles)
            print("found paired cycles: {}".format(', '.join(map(str, pcycles))))

            strokes = find_strokes(ends + [[len(merged) - 1, not ends[-1][1], True]], merged)

            support = sorted([(item[0], item[1].format(merged[stroke.start_idx].date, merged[stroke.end_idx].date))
                              for stroke in strokes for item in stroke.support if stroke.end_idx != len(merged) - 1], key=lambda x: x[0], reverse=True)
            pressure = sorted([(item[0], item[1].format(merged[stroke.start_idx].date, merged[stroke.end_idx].date))
                               for stroke in strokes for item in stroke.pressure if stroke.end_idx != len(merged) - 1], key=lambda x: x[0])

            f.write('current price: {}, {}\n'.format(klines[-1].date, klines[-1].close))

            f.write('support\n')
            for sup in support:
                f.write('{:0.2f},\t{:0.4f},\t{}\n'.format(sup[0], sup[0] / klines[-1].close, sup[1]))

            f.write('pressure\n')
            for pre in pressure:
                f.write('{:0.2f},\t{:0.4f},\t{}\n'.format(pre[0], pre[0] / klines[-1].close, pre[1]))

            f.write('-------------\n\n')
            f.flush()

