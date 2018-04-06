import json
import random
import datetime
import numpy as np

import train_test_data
from trade_board import index_function as infc

parameters = ['first', 'last', 'high', 'low', 'rsi', 'macd', 'signal_line', 'SMIsignal', 'emasignal', 'amount', 'assets']
currency = ['btc', 'bch', 'eth', 'etc', 'xrp', 'qtum', 'ltc', 'iota']
candle_state = [1, 3, 5, 15, 30]

input_size = len(parameters)    # rsi, emr12, emr26, macd, signal_line
timeline = 10
timestamp = 155665800   # default

log_point = 0

file_list = train_test_data.file_list
test_file = train_test_data.test_file



def load_data(currency, file_list, test_file):

    global train_data_dic, test_data_dic    # list feature -> dict feature

    #train_data = []
    #test_data = []

    train_data_dic = dict()
    test_data_dic = dict()

    for file_idx in range(len(currency)):
        coin_label = currency[file_idx]
        coin_currency = file_list.get(coin_label)

        train_dict_by_coin = []

        for file_name in coin_currency:
            with open('./currency_data/'+file_name, 'r') as file:
                data = json.load(file)

                for idx in range(len(data)):
                    train_dict_by_coin.append(data[idx])

        train_data_dic[coin_label] = train_dict_by_coin
        print(len(train_data_dic[coin_label]))

    for file_idx in range(len(currency)):
        coin_label = currency[file_idx]
        cur = test_file.get(coin_label)

        test_dict_by_coin = []

        for file_name in cur:
            with open('./currency_data/'+file_name, 'r') as file:
                data = json.load(file)

                for idx in range(len(data)):
                    test_dict_by_coin.append(data[idx])

        test_data_dic[coin_label] = test_dict_by_coin
        print(len(test_data_dic[coin_label]))

start = random.choice(range(timeline, 25000))
index = start
seed_money = 5000000
coin_holding = {'btc': 0.0,
                'bch': 0.0,
                'eth': 0.0,
                'etc': 0.0,
                'xrp': 0.0,
                'qtum': 0.0,
                'ltc': 0.0,
                'iota': 0.0}
coin_assets = {'btc': 0.0,
               'bch': 0.0,
               'eth': 0.0,
               'etc': 0.0,
               'xrp': 0.0,
               'qtum': 0.0,
               'ltc': 0.0,
               'iota': 0.0}
risk = {'btc': 1.0,
        'bch': 1.0,
        'eth': 1.0,
        'etc': 1.0,
        'xrp': 1.0,
        'qtum': 1.0,
        'ltc': 1.0,
        'iota': 1.0}
#coin_series = [[0]*len(currency) for _ in range(index)]   # time series: amount of stock holding
whole_assets = 5000000
past_assets = 5000000
test_state = False
empty_action = False


def reset(is_test=False):
    global seed_money, start, index, coin_holding, whole_assets, past_assets, test_state, risk, coin_assets
    if is_test:
        start = random.choice(range(1000, 5000))
    else:
        start = random.choice(range(1000, 35000))
    index = start
    coin_holding = {'btc': 0.0,
                    'bch': 0.0,
                    'eth': 0.0,
                    'etc': 0.0,
                    'xrp': 0.0,
                    'qtum': 0.0,
                    'ltc': 0.0,
                    'iota': 0.0}
    coin_assets = {'btc': 0.0,
                   'bch': 0.0,
                   'eth': 0.0,
                   'etc': 0.0,
                   'xrp': 0.0,
                   'qtum': 0.0,
                   'ltc': 0.0,
                   'iota': 0.0}
    risk = {'btc': 1.0,
            'bch': 1.0,
            'eth': 1.0,
            'etc': 1.0,
            'xrp': 1.0,
            'qtum': 1.0,
            'ltc': 1.0,
            'iota': 1.0}

    seed_money = 5000000
    whole_assets = 5000000
    past_assets = 5000000
    test_state = is_test

    # data shape: [currency length * data length]
    if test_state:
        cur_whole = dict()
        for coin_label in currency:
            cur_whole[coin_label] = test_data_dic[coin_label][max(0, index - (50*max(candle_state))):index]  # make candle chart later
    else:
        cur_whole = dict()
        for coin_label in currency:
            cur_whole[coin_label] = train_data_dic[coin_label][max(0, index - (50*max(candle_state))):index]


    state = []
    # exert coin amount on initial state of currency data
    for coin_label in currency:

        for minute in candle_state:
            one_coin_state = []
            chart_data = infc.candle_chart(cur_whole[coin_label], minute)

            for data in chart_data:      # 10 time series data(dict)
                one_frame_state = []
                price_data = []

                for col in parameters:
                    if col == 'first' and data[col] is not None:
                        price_data.append(data[col])
                    elif col == 'last' and data[col] is not None:
                        price_data.append(data[col])
                    elif col == 'high' and data[col] is not None:
                        price_data.append(data[col])
                    elif col == 'low' and data[col] is not None:
                        price_data.append(data[col])

                price_mean = np.mean(price_data)
                price_std = np.std(price_data)

                for col in parameters:
                    if col == 'first' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    elif col == 'last' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    elif col == 'high' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    elif col == 'low' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    else:
                        one_frame_state.append(data[col])

                one_coin_state.append(one_frame_state)

            state.append(one_coin_state)

    #print(np.shape(state))  # (10, 11, 40)
    return state


def step(action):
    global seed_money, start, index, coin_holding, whole_assets, past_assets, risk, empty_action, timestamp, log_point
    reward = 0
    terminal = False

    if test_state:
        data = test_data_dic

    else:
        data = train_data_dic

    try:
        index += 30                         # 30 minutes term
        _cur = data['btc'][index]
        del _cur                            # index error 때문에 집어넣음 ㅇㅇ
        cur_whole = dict()
        for coin_label in currency:
            cur_whole[coin_label] = data[coin_label][max(0, index - (50*max(candle_state))):index]

        if index - start >= 30000 and not test_state:
            terminal = True

        elif index - start >= 5000 and test_state:
            terminal = True

    except IndexError:
        terminal = True
        index -= 30
        cur_whole = dict()
        for coin_label in currency:
            cur_whole[coin_label] = data[coin_label][max(0, index - (50*max(candle_state))):index - 15]

    else:
        #print('index: %d' % index)
        if action_apply(action)[1] == 0:
            reward -= int(0.001 * seed_money)
            empty_action = True

        else:
            coin_label = currency[action_apply(action)[0]]
            ratio = action_apply(action)[1]

            # 0.999 means charge of each dealing for market(0.1%)
            if ratio > 0:   # ratio is adjusted as terms of 10%(10% - 100%)
                if seed_money > 0:
                    coin_holding[coin_label] += 0.999 * ratio * seed_money / float(cur_whole[coin_label][-1]['last'])
                    coin_holding[coin_label] = round(coin_holding[coin_label], 8)
                    seed_money = int(seed_money * (1.0 - (1 / risk[coin_label]) * ratio))
                    coin_assets[coin_label] = coin_holding[coin_label] * float(cur_whole[coin_label][-1]['last'])
                    timestamp = int(cur_whole[coin_label][-1]['timestamp'])
                    empty_action = False

                else:
                    empty_action = True

            else:           # ratio < 0 : sell
                if coin_holding[coin_label] > 0.0:
                    seed_money = int(seed_money - max(risk[coin_label] * ratio, -1.0) * 0.999 * \
                                                  coin_holding[coin_label] * float(cur_whole[coin_label][-1]['last']))
                    coin_holding[coin_label] = coin_holding[coin_label] * (1.0 + max(risk[coin_label] * ratio, -1.0))
                    coin_holding[coin_label] = round(coin_holding[coin_label], 8)
                    coin_assets[coin_label] = coin_holding[coin_label] * float(cur_whole[coin_label][-1]['last'])
                    timestamp = int(cur_whole[coin_label][-1]['timestamp'])
                    empty_action = False

                else:
                    reward -= int(0.001 * seed_money)
                    empty_action = True

        whole_assets = seed_money + int(sum([coin_holding[cur] * float(cur_whole[cur][-1]['last']) for cur in currency]))
        reward += whole_assets - past_assets

        past_assets = whole_assets

    if (index - start) % 25000 < 401 and (index - start) % 25000 > 370:   # randomly chosen point to show status
        #print(index, start)
        print(coin_holding)
        print(coin_assets)
        present_time = str(datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
        print('time: %s\twhole assets: %d\treward: %d' % (present_time, whole_assets, reward))

    if empty_action == False:
        log_point += 1
        log_index = str(int(log_point / 2000) + 1)
        with open('./trade_log/trade_log'+log_index+'.csv', 'a') as file:
            file.write(str(datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')))
            file.write('\t')
            file.write(str(coin_holding))
            file.write('\t')
            file.write(str(seed_money))
            file.write('\t')
            file.write(str(whole_assets))
            file.write('\n')

#    if empty_action == False:
#        print('%s\t%s\t%s\t%s' % (str(datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')),
#                                  str(coin_holding),
#                                  str(seed_money),
#                                  str(whole_assets)))

    state = []

    for cur in currency:
        cur_whole[cur][-1]['amount'] = coin_holding[cur]
        cur_whole[cur][-1]['assets'] = coin_assets[cur]

        for minute in candle_state:
            one_coin_state = []
            chart_data = infc.candle_chart(cur_whole[cur], minute)

            for data in chart_data:
                one_frame_state = []
                price_data = []

                for col in parameters:
                    if col == 'first' and data[col] is not None:
                        price_data.append(data[col])
                    elif col == 'last' and data[col] is not None:
                        price_data.append(data[col])
                    elif col == 'high' and data[col] is not None:
                        price_data.append(data[col])
                    elif col == 'low' and data[col] is not None:
                        price_data.append(data[col])

                price_mean = np.mean(price_data)
                price_std = np.std(price_data)

                for col in parameters:
                    if col == 'first' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    elif col == 'last' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    elif col == 'high' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    elif col == 'low' and data[col] is not None:
                        if price_std == 0:
                            one_frame_state.append(0)
                        else:
                            one_frame_state.append(int((data[col] - price_mean) / price_std * 1000))
                    else:
                        one_frame_state.append(data[col])

                one_coin_state.append(one_frame_state)
            state.append(one_coin_state)

    return state, reward, terminal, None

