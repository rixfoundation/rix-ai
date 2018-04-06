def rsi_by_minute(data, n):
    if len(data) < n:
        return None

    else:
        real_data = data[-n:]
        up = 0
        down = 0
        for idx in range(len(real_data) - 1):
            if real_data[idx]['last'] is None or real_data[idx + 1]['last'] is None:
                pass
            elif real_data[idx]['last'] > real_data[idx + 1]['last']:
                down += int(real_data[idx + 1]['last']) - int(real_data[idx]['last'])
            elif real_data[idx]['last'] < real_data[idx + 1]['last']:
                up += int(real_data[idx + 1]['last']) - int(real_data[idx]['last'])
            else:
                pass

        if up == down:
            #print([up, down])
            return 50
        else:
            return int(round((up / (up - down)) * 100, 0))


def ema_by_minute(data, n):
    variable = 'ema' + str(n)
    if len(data) < n:
        return None

    elif len(data) == n:
        real_data = data[-n:]
        close_value = 0
        for idx in range(len(real_data)):
            if real_data[idx]['last'] is not None:
                close_value += int(real_data[idx]['last'])

        return int(round(close_value / len(real_data), 0))

    else:
        return int(round(int(data[-1]['last']) * (2.0 / (n + 1)) + float(data[-2][variable]) * (1.0 - 2.0 / (n + 1)), 0))


def macd(data):
    if data[-1]['ema12'] is None or data[-1]['ema26'] is None:
        return None
    else:
        return data[-1]['ema12'] - data[-1]['ema26']


def signal_line(data, n):
    if len(data) < n or data[-n]['macd'] is None:
        return None

    elif data[-2]['signal_line'] is None:
        real_data = data[-n:]
        signal = 0
        for idx in range(len(real_data)):
            signal += float(real_data[idx]['macd'])
        return int(round(signal / len(real_data), 0))

    else:
        return int(round(float(data[-1]['macd']) * (2 / (n + 1)) + float(data[-2]['signal_line']) * (1 - 2 / (n + 1)), 0))


def difference(data):
    if len(data) < 10:  # standard: 10
        return None, None

    else:
        real_data = data[-10:]
        currency_last = []

        for idx in range(len(real_data)):
            if real_data[idx]['last'] is not None:
                currency_last.append(real_data[idx]['last'])

        if len(currency_last) == 0:
            diff = 0
            rdiff = 0

        else:
            hh = max(currency_last)
            ll = min(currency_last)
            diff = hh - ll
            rdiff = currency_last[-1] - (hh + ll) / 2

        return diff, rdiff


def single_avg(data):
    if len(data) < 12:
        return None, None
    else:
        single_avgdiff = (data[-1]['diff'] + data[-2]['diff'] + data[-3]['diff']) / 3
        single_avgrel = (data[-1]['rdiff'] + data[-2]['rdiff'] + data[-3]['rdiff']) / 3
        return int(round(single_avgdiff, 0)), int(round(single_avgrel, 0))


def double_avg(data):
    if len(data) < 14:
        return None
    else:
        double_avgdiff = (data[-1]['single_avgdiff'] + data[-2]['single_avgdiff'] + data[-3]['single_avgdiff']) / 3
        double_avgrel = (data[-1]['single_avgrel'] + data[-2]['single_avgrel'] + data[-3]['single_avgrel']) / 3
        if double_avgdiff == 0:
            return 0
        else:
            SMI = double_avgrel / (double_avgdiff / 2) * 100
            # return double_avgdiff, double_avgrel, SMI
            return int(round(SMI, 0))


def stch_mtm(data, n):  # we will use the case n=3, n=10
    if len(data) < 13 + n:
        return None
    else:
        signal = 0
        for idx in range(n):
            signal += data[-(idx + 1)]['SMI']
        return int(round(signal / n, 0))


def candle_chart(data, minute):
    if len(data) >= 50 * minute:
        length = 50
        data = data[-length * minute:]
        whole_chart = []
        for idx in range(length):
            price_list = []
            single_candle = dict()

            for subidx in range(minute):
                price_list.append(int(data[idx * minute + subidx]['last']))

            single_candle['first'] = price_list[0]
            single_candle['last'] = price_list[-1]
            single_candle['high'] = max(price_list)
            single_candle['low'] = min(price_list)

            whole_chart.append(single_candle)

    elif len(data) >= 20 * minute:
        length = int(len(data) / minute)
        data = data[-length * minute:]
        whole_chart = []
        for idx in range(length):
            price_list = []
            single_candle = dict()

            for subidx in range(minute):
                price_list.append(int(data[idx * minute + subidx]['last']))

            single_candle['first'] = price_list[0]
            single_candle['last'] = price_list[-1]
            single_candle['high'] = max(price_list)
            single_candle['low'] = min(price_list)

            whole_chart.append(single_candle)

    else:
        length = int(len(data) / minute)
        data = data[-length * minute:]
        whole_chart = []

        for idx in range(20 - length):
            single_candle = dict()
            single_candle['first'] = None
            single_candle['last'] = None
            single_candle['high'] = None
            single_candle['low'] = None

            whole_chart.append(single_candle)

        for idx in range(length):
            price_list = []
            single_candle = dict()

            for subidx in range(minute):
                price_list.append(int(data[idx * minute + subidx]['last']))

            single_candle['first'] = price_list[0]
            single_candle['last'] = price_list[-1]
            single_candle['high'] = max(price_list)
            single_candle['low'] = min(price_list)

            whole_chart.append(single_candle)

    #print(whole_chart)

    for idx in range(len(whole_chart)):
        whole_chart[idx]['rsi'] = rsi_by_minute(whole_chart[:idx + 1], 14)
        whole_chart[idx]['ema12'] = ema_by_minute(whole_chart[:idx + 1], 12)
        whole_chart[idx]['ema26'] = ema_by_minute(whole_chart[:idx + 1], 26)

        whole_chart[idx]['macd'] = macd(whole_chart[:idx + 1])
        whole_chart[idx]['signal_line'] = signal_line(whole_chart[:idx + 1], 9)
        whole_chart[idx]['diff'], whole_chart[idx]['rdiff'] = difference(whole_chart[:idx + 1])
        whole_chart[idx]['single_avgdiff'], whole_chart[idx]['single_avgrel'] = single_avg(whole_chart[:idx + 1])
        whole_chart[idx]['SMI'] = double_avg(whole_chart[:idx + 1])
        whole_chart[idx]['SMIsignal'] = stch_mtm(whole_chart[:idx + 1], 3)
        whole_chart[idx]['emasignal'] = stch_mtm(whole_chart[:idx + 1], 10)
        whole_chart[idx]['amount'] = 0  # initial value
        whole_chart[idx]['assets'] = 0  # initial value

    return whole_chart[-10:]
