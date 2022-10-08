import json
import matplotlib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from datetime import timedelta
from pandas import json_normalize

pd.options.display.float_format = '{:.2f}'.format
matplotlib.rc('font', family='BIZ UDGothic')

# APIキーの設定
api_Key = 'i3UxcsmVeywcxTtN/BHupD+yeuX+2Err'
secretKey = 'KAQ061hBTAEZIULKIiaovrzOplN5v4vGgc74D3iaRvzjUNGsCj5+GidkbGUkqNZp'



###---------------------------------------WaveNet---------------------------------------###
def get_data(symbol='BTC', interval='1hour', date=''):
    '''24時間分の1時間足データを取得'''
    endPoint = 'https://api.coin.z.com/public'
    path     = f'/v1/klines?symbol={symbol}&interval={interval}&date={date}'

    response = requests.get(endPoint + path)
    r = json.dumps(response.json(), indent=2)
    r2 = json.loads(r)
    df = json_normalize(r2['data'])
    if len(df):
        date = []
        for i in df['openTime']:
            i = int(i)
            tsdate = int (i / 1000)
            loc = datetime.utcfromtimestamp(tsdate)
            date.append(loc)
        df.index = date
        df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Tokyo')
        df.drop('openTime', axis=1, inplace=True)
    return df

def get_today():
    '''YYYYMMDD形式で現在の日付を取得する'''
    now_time = datetime.now()
    current_year = now_time.year
    current_month = now_time.month
    current_day = now_time.day
    if current_month >= 10 and current_day >= 10:
        today = str(current_year) + str(current_month) + str(current_day)
    elif current_month < 10 and current_day >= 10:
        today = str(current_year) + '0' + str(current_month) + str(current_day)
    elif current_month < 10 and current_day < 10:
        today = str(current_year) + '0' + str(current_month) + '0' + str(current_day)
    elif current_month >= 10 and current_day < 10:
        today = str(current_year) + str(current_month) + '0' + str(current_day)
    return today

def get_train_data():
    '''学習用データ取得'''
    day = get_today()
    if datetime.now().hour > 6:
        btc_today = get_data(symbol='BTC_JPY', interval='1hour', date=day)
        eth_today = get_data(symbol='ETH_JPY', interval='1hour', date=day)
    else:
        day = datetime.strptime(day, '%Y%m%d')
        day -= timedelta(days=1)
        day = str(day)
        day = day.replace('-', '')
        day = day.replace(' 00:00:00', '')
        btc_today = get_data(symbol='BTC_JPY', interval='1hour', date=day)
        eth_today = get_data(symbol='ETH_JPY', interval='1hour', date=day)
    if len(eth_today) == 0:
        eth_today = pd.DataFrame(data=np.array([[None for i in range(24)] for i in range(5)]).T, index=btc_today.index, columns=['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume'])
    eth_today.columns = ['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume']
    df = pd.concat([btc_today, eth_today], axis=1)
    for i in range(480):
        day = datetime.strptime(day, '%Y%m%d')
        day -= timedelta(days=1)
        day = str(day)
        day = day.replace('-', '')
        day = day.replace(' 00:00:00', '')
        btc = get_data(symbol='BTC_JPY', interval='1hour', date=day)
        eth = get_data(symbol='ETH_JPY', interval='1hour', date=day)
        if len(eth) == 0:
            eth = pd.DataFrame(data=np.array([[None for i in range(24)] for i in range(5)]).T, index=btc.index, columns=['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume'])
        eth.columns = ['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume']
        tmp_df = pd.concat([btc, eth], axis=1)
        df = pd.concat([tmp_df, df], axis=0)
    return df