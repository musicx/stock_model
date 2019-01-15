import pandas as pd
import QUANTAXIS as qa
import datetime as dt


def up_stage(data, col):
    rate = data[col] / (data[col].shift(1)) - 1
    six = (rate > 0.06) * 1
    eight = (rate > 0.08) * 1
    ten = (rate > 0.097) * 1
    return pd.DataFrame({'cat': six+eight+ten, 'rate': rate})


if __name__ == '__main__':
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    today_str = today.strftime('%Y-%m-%d')

    # divide_date = today - dt.timedelta(days=60)
    # divide_str = divide_date.strftime('%Y-%m-%d')

    stocks = qa.QA_fetch_stock_list_adv()
    stock_list = stocks.code.tolist()
    # stock_list = ZZ800.split('\n')
    blocks = []
    for stock in stock_list:
        print('%s: handling %s' % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stock))
        try:
            block = qa.QA_fetch_stock_block_adv(stock)
        except:
            print('data error during {}'.format(stock))
            continue
        block_names = block.data.reset_index().loc[:, ['code', 'blockname']]
        blocks.append(block_names)
    block_name = pd.concat(blocks, axis=0)

    candles = qa.QA_fetch_stock_day_adv(stock_list, start='2019-01-01', end='2019-01-14').to_qfq()
    stage = candles.add_func(up_stage, 'close')
