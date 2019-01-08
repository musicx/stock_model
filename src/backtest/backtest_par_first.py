from pprint import pprint as print
from QUANTAXIS.QAARP.QARisk import QA_Risk
from QUANTAXIS.QAARP.QAUser import QA_User
from QUANTAXIS.QABacktest.QABacktest import QA_Backtest
from QUANTAXIS.QAUtil.QALogs import QA_util_log_info


from QUANTAXIS.QAARP.QAStrategy import QA_Strategy
from QUANTAXIS.QAUtil.QAParameter import (AMOUNT_MODEL, MARKET_TYPE,
                                          FREQUENCE, ORDER_DIRECTION,
                                          ORDER_MODEL)


class MAStrategy(QA_Strategy):
    def __init__(self):
        super().__init__()
        self.frequence = FREQUENCE.DAY
        self.market_type = MARKET_TYPE.STOCK_CN
        self.commission_coeff = 0.00015
        self.tax_coeff = 0.0001

    def on_bar(self, event):
        sellavailable = self.sell_available
        try:
            for item in event.market_data.code:
                if sellavailable is None:

                    event.send_order(account_id=self.account_cookie,
                                     amount=100, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                     time=self.current_time, code=item, price=item.high[0],
                                     order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                     market_type=self.market_type, frequence=self.frequence,
                                     broker_name=self.broker)

                else:
                    if sellavailable.get(item, 0) > 0:
                        event.send_order(account_id=self.account_cookie,
                                         amount=sellavailable[item], amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                         time=self.current_time, code=item, price=0,
                                         order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.SELL,
                                         market_type=self.market_type, frequence=self.frequence,
                                         broker_name=self.broker
                                         )
                    else:
                        event.send_order(account_id=self.account_cookie,
                                         amount=100, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                         time=self.current_time, code=item, price=0,
                                         order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                         market_type=self.market_type, frequence=self.frequence,
                                         broker_name=self.broker)
        except:
            pass


class MAMINStrategy(QA_Strategy):
    def __init__(self):
        super().__init__()
        self.frequence = FREQUENCE.FIFTEEN_MIN
        self.market_type = MARKET_TYPE.STOCK_CN

    def on_bar(self, event):
        try:
            for item in event.market_data.code:
                if self.sell_available is None:
                    print(self.current_time)

                    event.send_order(account_id=self.account_cookie,
                                     amount=10000, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                     time=self.current_time, code=item, price=0,
                                     order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                     market_type=self.market_type, frequence=self.frequence,
                                     broker_name=self.broker)

                else:
                    if self.sell_available.get(item, 0) > 0:
                        event.send_order(account_id=self.account_cookie,
                                         amount=self.sell_available[item], amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                         time=self.current_time, code=item, price=0,
                                         order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.SELL,
                                         market_type=self.market_type, frequence=self.frequence,
                                         broker_name=self.broker
                                         )
                    else:
                        event.send_order(account_id=self.account_cookie,
                                         amount=10000, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                         time=self.current_time, code=item, price=0,
                                         order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                         market_type=self.market_type, frequence=self.frequence,
                                         broker_name=self.broker)
        except:
            pass


class DUOStrategy(QA_Strategy):
    def __init__(self):
        super().__init__()

    def on_bar(self, event):
        if self.market_data.len > 1:
            for item in event.market_data.code:
                if self.sell_available is None:
                    event.send_order(account_id=self.account_cookie,
                                     amount=10000, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                     time=self.current_time, code=item, price=0,
                                     order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                     market_type=MARKET_TYPE.STOCK_CN, frequence=FREQUENCE.DAY,
                                     broker_name=self.broker)

                else:
                    if self.sell_available.get(item, 0) > 0:
                        event.send_order(account_id=self.account_cookie,
                                         amount=self.sell_available[item], amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                         time=self.current_time, code=item, price=0,
                                         order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.SELL,
                                         market_type=MARKET_TYPE.STOCK_CN, frequence=FREQUENCE.DAY,
                                         broker_name=self.broker
                                         )
                    else:
                        event.send_order(account_id=self.account_cookie,
                                         amount=10000, amount_model=AMOUNT_MODEL.BY_AMOUNT,
                                         time=self.current_time, code=item, price=0,
                                         order_model=ORDER_MODEL.MARKET, towards=ORDER_DIRECTION.BUY,
                                         market_type=MARKET_TYPE.STOCK_CN, frequence=FREQUENCE.DAY,
                                         broker_name=self.broker)


class Backtest(QA_Backtest):
    '''
    多线程模式回测示例

    '''

    def __init__(self, market_type, frequence, start, end, code_list, commission_fee):
        super().__init__(market_type,  frequence, start, end, code_list, commission_fee)
        self.user = QA_User()
        mastrategy = MAStrategy()
        # maminstrategy = MAMINStrategy()
        # maminstrategy.reset_assets(1000)
        # self.portfolio, self.account = self.user.register_account(mastrategy)
        self.user = QA_User(user_cookie='user_admin')
        self.portfolio = self.user.new_portfolio('folio_admin')
        self.portfolio, self.account = self.user.register_account(mastrategy)

    def after_success(self):
        QA_util_log_info(self.account.history_table)
        risk = QA_Risk(self.account, benchmark_code='000300',
                       benchmark_type=MARKET_TYPE.INDEX_CN)

        print(risk().T)

        self.account.save()
        risk.save()


def run_daybacktest():
    import QUANTAXIS as QA
    backtest = Backtest(market_type=MARKET_TYPE.STOCK_CN,
                        frequence=FREQUENCE.DAY,
                        start='2017-01-01',
                        end='2017-02-10',
                        code_list=QA.QA_fetch_stock_block_adv().code[0:5],
                        commission_fee=0.00015)
    backtest.start_market()

    backtest.run()
    backtest.stop()


def run_minbacktest():
    import QUANTAXIS as QA
    backtest = Backtest(market_type=MARKET_TYPE.STOCK_CN,
                        frequence=FREQUENCE.FIFTEEN_MIN,
                        start='2017-11-01',
                        end='2017-11-10',
                        code_list=QA.QA_fetch_stock_block_adv().code[0:5],
                        commission_fee=0.00015)
    backtest.start_market()

    backtest.run()
    backtest.stop()


if __name__ == '__main__':
    run_daybacktest()
    # run_minbacktest()
    # backtest._settle()

    # backtest.run()
