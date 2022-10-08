import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from data.make_data import get_train_data
from features.build_features import data_cleaning, feature_engineering, feature_engineering_lgbm, feature_engineering_lr
from models.models import mish, wavenet_training


##------------------データの取得------------------##
df = get_train_data()
df = data_cleaning(df)
df.to_csv('./data/data.csv')

##------------------WaveNet------------------##
# データ分割・特徴量生成
df = pd.read_csv('./data/data.csv', index_col='Unnamed: 0')
df = df.head(len(df)-24)
x_data, y_data = feature_engineering(df)
x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, shuffle=False)
# モデル構築・保存
model = wavenet_training(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)
tf.keras.models.save_model(model, './models/wavenet.h5')

##------------------Logistic------------------##
df = pd.read_csv('./data/data.csv', index_col='Unnamed: 0')
df = df.head(len(df)-24)
X, y = feature_engineering_lr(df)
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_valid)
lr = LogisticRegression()
lr.fit(x_train, y_train)
with open('./models/logistic.pickle', mode='wb') as f:
    pickle.dump(lr, f)
    

lr = LogisticRegression()
lr.fit(x_train, y_train)

##------------------Light GBM------------------##
df = pd.read_csv('./data/data.csv', index_col='Unnamed: 0')
df = df.head(len(df)-24)
X, y = feature_engineering_lgbm(df)
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
gbm = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', learning_rate=0.1, max_depth=-1,
              min_child_samples=30, min_child_weight=0.001, min_split_gain=0.0,
              n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
              random_state=0, reg_alpha=0.0, reg_lambda=0.0, silent=True,
              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
eval_set = [(x_valid, y_valid)]
callbacks = []
callbacks.append(lgb.early_stopping(stopping_rounds=10))
callbacks.append(lgb.log_evaluation())
gbm.fit(x_train, y_train, eval_set=eval_set, callbacks=callbacks)
with open('./models/lgbm.pickle', mode='wb') as f:
    pickle.dump(gbm, f)