import numpy as np
import pandas as pd
import QUANTAXIS as qa


def higher(x, y):
    return x[2] >= y[2] and x[3] >= y[3]


def contain(x, y):
    return x[2] >= y[2] and x[3] <= y[3]


def lower(x, y):
    return x[2] <= y[2] and x[3] <= y[3]


def merge_ochl(ochl_list, times):
    merged = []
    merged_cnt = []
    merged_last_times = []
    times = times.tolist()
    merge_idx = 0

    for raw_idx, ochl in enumerate(ochl_list):
        if raw_idx == 0:
            merged.append(ochl_list[raw_idx])
            merged_last_times.append(times[raw_idx])
            merged_cnt.append(1)
            continue
        if contain(merged[merge_idx], ochl) or contain(ochl, merged[merge_idx]):
            if merge_idx == 0 or merged[merge_idx-1][2] <= merged[merge_idx][2]:
                merged[merge_idx][0] = max(merged[merge_idx][0], ochl[0])
                merged[merge_idx][1] = max(merged[merge_idx][1], ochl[1])
                merged[merge_idx][2] = max(merged[merge_idx][2], ochl[2])
                merged[merge_idx][3] = max(merged[merge_idx][3], ochl[3])
            else:
                merged[merge_idx][0] = min(merged[merge_idx][0], ochl[0])
                merged[merge_idx][1] = min(merged[merge_idx][1], ochl[1])
                merged[merge_idx][2] = min(merged[merge_idx][2], ochl[2])
                merged[merge_idx][3] = min(merged[merge_idx][3], ochl[3])
            merged_last_times[merge_idx] = times[raw_idx]
            merged_cnt[merge_idx] += 1
        else:
            merged.append(ochl)
            merged_last_times.append(times[raw_idx])
            merged_cnt.append(1)
            merge_idx += 1

    return merged, merged_last_times, merged_cnt


def find_endpoints(merged_ochl_list, merged_count):
    end = []  # item is (idx in merged list, is top end point, valid end point)
    last = 0
    for idx, ochl in enumerate(merged_ochl_list):
        if idx < 2:
            continue
        insert = False
        if higher(merged_ochl_list[idx - 1], merged_ochl_list[idx - 2]) and higher(merged_ochl_list[idx - 1], ochl):
            end.append([idx - 1, True, True])
            insert = True
        elif lower(merged_ochl_list[idx - 1], merged_ochl_list[idx - 2]) and lower(merged_ochl_list[idx - 1], ochl):
            end.append([idx - 1, False, True])
            insert = True

        if insert and len(end) > 1 and end[-1][1] == end[-2][1]:
            if ((end[-1][1] and merged_ochl_list[end[-1][0]][2] > merged_ochl_list[end[-2][0]][2]) or
                    (not end[-1][1] and merged_ochl_list[end[-1][0]][3] < merged_ochl_list[end[-2][0]][3])):
                if not (idx - 2 == last and merged_count[last] + merged_count[last + 1] > 2 or idx - 3 > last):
                    end.pop(-2)
                else:
                    end[-2][2] = False
            else:
                end.pop()
                insert = False
        elif insert and not (idx - 2 == last and merged_count[last] + merged_count[last + 1] > 2 or idx - 3 > last):
            end.pop()
            insert = False

        if insert:
            last = idx
    return end


class Stroke(object):
    def __init__(self, start_idx, end_idx, ochl):
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.is_bull = ochl[start_idx][2] < ochl[end_idx][2] and ochl[start_idx][3] < ochl[end_idx][3]
        self.high = max(ochl[start_idx][2], ochl[end_idx][2])
        self.low = min(ochl[start_idx][3], ochl[end_idx][3])
        span = self.high - self.low
        self.low_energy = self.low + span * 0.382 if self.is_bull else self.high - span * 0.382
        self.mid_energy = self.low + span * 0.5
        self.high_energy = self.low + span * 0.382 if not self.is_bull else self.high - span * 0.382
        self.first_target = self.high + span * 0.618 if self.is_bull else self.low - span * 0.618
        self.second_target = self.high + span * 1.618 if self.is_bull else self.low - span * 1.618

        within = [(self.low_energy, '0.382 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                  (self.mid_energy,  '0.5   of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                  (self.high_energy, '0.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down'))]
        outside = [(self.high if self.is_bull else self.low, 'end   of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                   (self.first_target, '1.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                   (self.second_target, '2.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down'))]

        self.support = within if self.is_bull else outside
        self.pressure = outside if self.is_bull else within

        self.end_change = False
        self.valid = True

    def change_endpoint(self):
        if self.is_bull:
            try:
                self.pressure.remove((self.high, 'end of\t{} up to\t{}'))
            except:
                pass
            self.support.append((self.high, 'end of\t{} up to\t{}'))
        else:
            try:
                self.support.remove((self.low, 'end of\t{} down to\t{}'))
            except:
                pass
            self.pressure.append((self.low, 'end of\t{} down to\t{}'))

    def tostring(self):
        return '{} {} to {}'.format(self.start_idx, 'up' if self.is_bull else 'down', self.end_idx)


def find_strokes(points, merged_ochl_list):
    valid_points = [x for x in points if x[2]]
    strokes = [Stroke(valid_points[0][0], valid_points[1][0], merged_ochl_list)]

    for idx, point in enumerate(valid_points):
        if idx < 2:
            continue
        for stroke in strokes:
            if ((point[1] and ((merged_ochl_list[point[0]][1] > stroke.low_energy and not stroke.is_bull) or
                               (merged_ochl_list[point[0]][2] > stroke.second_target and stroke.is_bull))) or
                    (not point[1] and ((merged_ochl_list[point[0]][1] > stroke.low_energy and stroke.is_bull) or
                                       (merged_ochl_list[point[0]][3] > stroke.second_target and not stroke.is_bull)))):
                stroke.valid = False
                continue
            stroke.support = [p for p in stroke.support if p[0] < merged_ochl_list[point[0]][3]]
            stroke.pressure = [p for p in stroke.pressure if p[0] > merged_ochl_list[point[0]][2]]
            if (not stroke.end_change and
                    ((stroke.is_bull and merged_ochl_list[point[0]][1] > stroke.high) or
                     (not stroke.is_bull and merged_ochl_list[point[0]][1] < stroke.low))):
                stroke.end_change = True
                stroke.change_endpoint()

        strokes = [s for s in strokes if s.valid]

        names = set([s.tostring() for s in strokes])

        if point[1]:
            starts = [s.start_idx for s in strokes if s.is_bull == point[1] and s.high < merged_ochl_list[point[0]][2]]
        else:
            starts = [s.start_idx for s in strokes if s.is_bull == point[1] and s.low > merged_ochl_list[point[0]][3]]

        new_strokes = [Stroke(x, point[0], merged_ochl_list) for x in starts]
        new_strokes.append(Stroke(valid_points[idx - 1][0], point[0], merged_ochl_list))

        for stroke in new_strokes:
            if stroke.tostring() not in names:
                strokes.append(stroke)

    return strokes




if __name__ == '__main__':

    can = qa.QA_fetch_stock_day_adv('000002', start='2017-06-01', end='2018-05-30').to_qfq()
    raw = can.data.loc[:, ['open', 'close', 'high', 'low']].values
    dates = can.data.date

    merged, merged_dates, merged_cnt = merge_ochl(raw, dates)
    ends = find_endpoints(merged, merged_cnt)
    strokes = find_strokes(ends, merged)

    support = sorted([(item[0], item[1].format(merged_dates[stroke.start_idx], merged_dates[stroke.end_idx]))
                      for stroke in strokes for item in stroke.support], key=lambda x: x[0], reverse=True)
    pressure = sorted([(item[0], item[1].format(merged_dates[stroke.start_idx], merged_dates[stroke.end_idx]))
                       for stroke in strokes for item in stroke.pressure], key=lambda x: x[0])

    print('current price: {}, {}'.format(dates[-1], raw[-1]))

    print('support')
    for pre in support:
        print('{:0.2f},\t{}'.format(pre[0], pre[1]))

    print('pressure')
    for pre in pressure:
        print('{:0.2f},\t{}'.format(pre[0], pre[1]))
