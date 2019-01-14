import pandas as pd
import QUANTAXIS as qa
import datetime as dt


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
