from email import message
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle
from datetime import datetime
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, make_response
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data.make_data import get_train_data
from features.build_features import data_cleaning, feature_engineering, feature_engineering_lgbm, feature_engineering_lr
from models.models import mish, wavenet_training
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


# アプリの設定
app = Flask(__name__)



#################--------------3種類のモデル(Logistic, LGBM, WaveNet)を使って予測値を出す--------------#################
# 対数収益率と二値分類
# 1時間足のデータなので、24時間に一回予測するなら、24行後との比較で、対数収益率を算出する必要がある
#btc_price.dropna(inplace=True)
# 予測モデルを使用することで、収益率を24時間に一回予測する。
#################--------------3種類のモデル(Logistic, LGBM, WaveNet)を使って予測値を出す--------------#################


#################--------------ホームページ(予測したい日付とモデルを入れる)--------------#################
@app.route("/")
def index():
    # osモジュールでパスを指定
    cd_path = os.getcwd()
    data_pass = os.path.join(cd_path, 'data', 'data.csv')
    df = pd.read_csv(data_pass, index_col='Unnamed: 0')
    df = df.tail(1)
    update = df.index[0]
    update = update[:-6]
    update = update.replace(' ', '')
    update = datetime.strptime(update, '%Y-%m-%d%H:%M:%S')
    year = update.year
    month = update.month
    day = update.day
    hour = update.hour
    update_text = f'最終更新日: {year}年{month}月{day}日{hour}時'
    return render_template("index.html", update_text=update_text)
#################--------------ホームページ(予測したい日付とモデルを入れる)--------------#################


#################--------------予測結果を返すページ--------------#################
@app.route('/predict', methods=['POST'])
def predict():
    str_features = [x for x in request.form.values()]
    print(str_features)
    date = str_features[1] + "-"  + str_features[2] + "-" + str_features[3] + " " +  str_features[4] + ":00:00+09:00"
    selected_model = request.form['type']
    selected_model = str(selected_model)
    date_2 = date[:-6]
    date_2 = date_2.replace(' ', '')
    date_2 = datetime.strptime(date_2, '%Y-%m-%d%H:%M:%S')
    date_2 = date_2 - timedelta(hours=1)
    year = date_2.year
    month = date_2.month
    day = date_2.day
    hour = date_2.hour
    if month < 10:
        month = '0' + str(month)
    if day < 10:
        day = '0' + str(day)
    if hour < 10:
        hour = '0' + str(hour)
    date_2 = str(year) + '-' + str(month) + '-' + str(day) + ' ' + str(hour) + ':00:00+09:00'

    if selected_model == 'Wave Net':
        import tensorflow as tf
        from tensorflow.keras.utils import get_custom_objects
        get_custom_objects().update({'mish': mish})
        cd_path = os.getcwd()
        data_pass = os.path.join(cd_path, 'data', 'data.csv')
        df = pd.read_csv(data_pass, index_col='Unnamed: 0')
        if date_2 not in df.index:
            return render_template("index.html", caution_text='日時を再入力してください。')
        df = df.loc[df.index <= date_2]
        df.fillna(method='ffill')
        x_data, y_data = feature_engineering(df)
        x_test = x_data[-1, :, :]
        x_test = x_test[np.newaxis, :, :]
        cd_path = os.getcwd()
        model_pass = os.path.join(cd_path, 'models', 'wavenet.h5')
        model = tf.keras.models.load_model(model_pass, custom_objects={'mish': mish})
        prediction = model.predict(x_test)
        if prediction > 0:
            return render_template('index.html', 
                                    prediction_text=f'{date}のマーケットの方向はプラスです',
                                    selected_model_text=f'選択したモデルは{selected_model}です')
        else:
            return render_template('index.html', 
                                    prediction_text=f'{date}のマーケットの方向はマイナスです',
                                    selected_model_text=f'選択したモデルは{selected_model}です')
            
    elif selected_model == 'Light GBM':
        cd_path = os.getcwd()
        data_pass = os.path.join(cd_path, 'data', 'data.csv')
        df = pd.read_csv(data_pass, index_col='Unnamed: 0')
        if date_2 not in df.index:
            return render_template("index.html", caution_text='日時を再入力してください。')
        df = df.loc[df.index <= date_2]
        df.fillna(method='ffill')
        X, y = feature_engineering_lgbm(df)
        X_test = X.tail(1)
        cd_path = os.getcwd()
        model_pass = os.path.join(cd_path, 'models', 'lgbm.pickle')
        with open(model_pass, mode='rb') as f:  # with構文でファイルパスとバイナリ読み込みモードを設定
            gbm = pickle.load(f) 
        prediction = gbm.predict(X_test)
        if prediction > 0:
            return render_template('index.html', 
                                    prediction_text=f'{date}のマーケットの方向はプラスです',
                                    selected_model_text=f'選択したモデルは{selected_model}です')
        else:
            return render_template('index.html', 
                                    prediction_text=f'{date}のマーケットの方向はマイナスです',
                                    selected_model_text=f'選択したモデルは{selected_model}です')

    elif selected_model == 'Logistic Regression':
        cd_path = os.getcwd()
        data_pass = os.path.join(cd_path, 'data', 'data.csv')
        df = pd.read_csv(data_pass, index_col='Unnamed: 0')
        if date_2 not in df.index:
            return render_template("index.html", caution_text='日時を再入力してください。')
        df = df.loc[df.index <= date_2]
        df.fillna(method='ffill')
        X, y = feature_engineering_lr(df)
        X_test = X.tail(1)
        cd_path = os.getcwd()
        model_pass = os.path.join(cd_path, 'models', 'logistic.pickle')
        with open(model_pass, mode='rb') as f:  # with構文でファイルパスとバイナリ読み込みモードを設定
            lr = pickle.load(f)
        prediction = lr.predict(X_test)
        if prediction > 0:
            return render_template('index.html', 
                                    prediction_text=f'{date}のマーケットの方向はプラスです',
                                    selected_model_text=f'選択したモデルは{selected_model}です')
        else:
            return render_template('index.html', 
                                    prediction_text=f'{date}のマーケットの方向はマイナスです',
                                    selected_model_text=f'選択したモデルは{selected_model}です')
    else:
        return render_template('index.html', selected_model_text='モデルを選択してください')
#################--------------予測結果を返すページ--------------#################


#################--------------価格推移を表示するページ--------------#################
@app.route("/visualize_date")
def visualize_date():
    return render_template("visualize.html")

@app.route('/visualize', methods=['POST'])
def visualize():
    str_features = [x for x in request.form.values()]
    print(str_features)
    date = str_features[0] + "-"  + str_features[1] + "-" + str_features[2] + "-" +  str_features[3] + ":00:00+09:00"
    span = request.form['span']
    span = int(span)
    cd_path = os.getcwd()
    data_pass = os.path.join(cd_path, 'data', 'data.csv')
    df = pd.read_csv(data_pass, index_col='Unnamed: 0')
    df = df.loc[df.index <= date]
    if len(df) == 0:
        return render_template("visualize.html", message='日時を再入力してください。')
    df.fillna(method='ffill', inplace=True)
    df = df.astype(float)
    df = df.tail(span*24)
    fig = plt.figure(figsize=(20, 15))
    plt.plot(df.index, df['open'].values, label='open price')
    plt.xlabel('date')
    plt.ylabel('price (million yen)')
    plt.xticks(rotation=45)
    plt.title('BTC price')
    plt.legend()
    plt.grid()
    canvas = FigureCanvasAgg(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    data = png_output.getvalue()

    response = make_response(data)
    response.headers['Content-Type'] = 'image/png'
    
    return response


if __name__ == "__main__":
    app.run(debug=True)