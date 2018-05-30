import numpy as np
import pandas as pd
import QUANTAXIS as qa

can = qa.QA_fetch_stock_day_adv('000002', start='2017-06-01', end='2018-05-29').to_qfq()
raw = can.data.loc[:, ['open', 'close', 'high', 'low']].values
dates = can.data.date



mrg = []
mrg_cnt = []
mrg_date = []
raw = raw
raw_date = dates.tolist()
raw_idx = 0
mrg_idx = 0

while True:
    if raw_idx >= len(raw):
        break
    if raw_idx == 0:
        mrg.append(raw[raw_idx])
        mrg_date.append(raw_date[raw_idx])
        mrg_cnt.append(1)
        raw_idx += 1
        continue
    if ((mrg[mrg_idx][2] >= raw[raw_idx][2] and mrg[mrg_idx][3] <= raw[raw_idx][3]) or
            (mrg[mrg_idx][2] <= raw[raw_idx][2] and mrg[mrg_idx][3] >= raw[raw_idx][3])):
        if mrg_idx == 0 or mrg[mrg_idx-1][2] <= mrg[mrg_idx][2]:
            mrg[mrg_idx][0] = max(mrg[mrg_idx][0], raw[raw_idx][0])
            mrg[mrg_idx][1] = max(mrg[mrg_idx][1], raw[raw_idx][1])
            mrg[mrg_idx][2] = max(mrg[mrg_idx][2], raw[raw_idx][2])
            mrg[mrg_idx][3] = max(mrg[mrg_idx][3], raw[raw_idx][3])
        else:
            mrg[mrg_idx][0] = min(mrg[mrg_idx][0], raw[raw_idx][0])
            mrg[mrg_idx][1] = min(mrg[mrg_idx][1], raw[raw_idx][1])
            mrg[mrg_idx][2] = min(mrg[mrg_idx][2], raw[raw_idx][2])
            mrg[mrg_idx][3] = min(mrg[mrg_idx][3], raw[raw_idx][3])
        mrg_date[mrg_idx] = raw_date[raw_idx]
        mrg_cnt[mrg_idx] += 1
        raw_idx += 1
        continue
    mrg.append(raw[raw_idx])
    mrg_date.append(raw_date[raw_idx])
    mrg_cnt.append(1)
    raw_idx += 1
    mrg_idx += 1


def higher(x, y):
    return x[2] >= y[2] and x[3] >= y[3]


def lower(x, y):
    return x[2] <= y[2] and x[3] <= y[3]


end = []
last = 0
for idx, cell in enumerate(mrg):
    insert = False
    if idx < 2:
        continue
    if higher(mrg[idx - 1], mrg[idx - 2]) and higher(mrg[idx - 1], cell):
        end.append([idx - 1, 1, True])
        insert = True
    elif lower(mrg[idx - 1], mrg[idx - 2]) and lower(mrg[idx - 1], cell):
        end.append([idx - 1, 0, True])
        insert = True

    if insert and len(end) > 1 and end[-1][1] == end[-2][1]:
        if ((end[-1][1] == 1 and mrg[end[-1][0]][2] > mrg[end[-2][0]][2]) or
                (end[-1][1] == 0 and mrg[end[-1][0]][3] < mrg[end[-2][0]][3])):
            if not (idx - 2 == last and mrg_cnt[last] + mrg_cnt[last + 1] > 2 or idx - 3 > last) and len(end) > 2:
                end.pop(-2)
            else:
                end[-2][2] = False
        else:
            end.pop()
            insert = False
    elif insert and not (idx - 2 == last and mrg_cnt[last] + mrg_cnt[last + 1] > 2 or idx - 3 > last):
        end.pop()
        insert = False

    if insert:
        last = idx

valid = [x for x in end if x[2]]


class Line(object):
    def __init__(self, start_idx, end_idx, ochl):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.is_bull = ochl[start_idx][2] < ochl[end_idx][2] and ochl[start_idx][3] < ochl[end_idx][3]
        self.start = ochl[start_idx][3] if self.is_bull else ochl[start_idx][2]
        self.end = ochl[end_idx][2] if self.is_bull else ochl[end_idx][3]
        self.high = self.end if self.is_bull else self.start
        self.low = self.start if self.is_bull else self.end
        span = self.high - self.low
        self.low_energy = self.low + span * 0.382 if self.is_bull else self.high - span * 0.382
        self.mid_energy = self.low + span * 0.5
        self.high_energy = self.low + span * 0.382 if not self.is_bull else self.high - span * 0.382
        self.first_target = self.high + span * 0.618 if self.is_bull else self.low - span * 0.618
        self.second_target = self.high + span * 1.618 if self.is_bull else self.low - span * 1.618
        self.valid = True

