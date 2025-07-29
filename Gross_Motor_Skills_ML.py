Pythonコード

# ライブラリのインポート
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
import japanize_matplotlib  # Matplotlibで日本語を使用可能にする
from scipy.stats import mstats
import scipy.stats as stats
warnings.filterwarnings('ignore', category=UserWarning)
import lightgbm as lgb
import random



# シード値の設定
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# データの読み込み
df=pd.read_excel('df_all_2.xlsx')
df.columns

#参加同意者のみ取得
df = df.drop(df[(df['Consensus'] == 0)].index, axis=0)

#特徴量の抽出
df_2=df[[ 'sex',
         'age',
         'birth_month_2',
         'grit_total',
         'res_total',
         'int_total',
         'leisure_time_p1',
         'leisure_time_p2', 
         'leisure_time_p3', 
         'bro_sis',
         'sporthistory',
         'fa_abi'
         'mo_abi', 
       'sportkind_1yr', 
        'first_sportkind',
       'competitive_results', 
       'recognition_abi',
       'compliments',
       'aqui_abi', 
         'sub_abi'
]]

df_2.shape

#欠損値の削除
df_2=df_2.dropna(axis=0)
df_2.shape


# 特徴量と目的変数の設定
X = df_2.drop(['sub_abi'], axis=1)
y = df_2['sub_abi']


# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)



# カテゴリ変数のlabel encoding
from sklearn.preprocessing import LabelEncoder
cat_cols = ['leisure_time_p1',
            'leisure_time_p2', 
            'leisure_time_p3',
            'bro_sis',
            'sporthistory',
            'competitive_results',
            'compliments',
            'aqui_abi'
           ]



for c in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[c])
    X_train[c] = le.transform(X_train[c])
    X_test[c] = le.transform(X_test[c])


X_train.info() 
X_train.shape



# カテゴリ変数のデータ型をcategory型に変換
cat_cols = ['leisure_time_p1',
            'leisure_time_p2', 
            'leisure_time_p3',
            'bro_sis',

            'sporthistory',
            'competitive_results',
            'compliments',
            'aqui_abi'
           ]



for c in cat_cols:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

X_train.info()



# 学習データの一部を検証データに分割
X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=123)
print('X_trの形状：', X_tr.shape, ' y_trの形状：', y_tr.shape, ' X_vaの形状：', X_va.shape, ' y_vaの形状：', y_va.shape)



# optunaのバージョン確認
import optuna
optuna.__version__

# 固定値のハイパーパラメータ
params_base = {
    'objective': 'mae'
    'random_seed': 1234,
    'learning_rate': 0.01,
    'min_data_in_bin': 3,
    'bagging_freq': 1
    'bagging_seed': 0,
    'verbose': -1,
}

     

# ハイパーパラメータ最適化
# ハイパーパラメータの探索範囲
def objective(trial):
    params_tuning = {
      'num_leaves': trial.suggest_int('num_leaves', 50, 200),
      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 30),
      'max_bin': trial.suggest_int('max_bin', 200, 400),
      'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 0.95),
      'feature_fraction': trial.suggest_float('feature_fraction', 0.35, 0.65),
      'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 1, log=True),
      'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 1, log=True),
      'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 1, log=True),
  }

  

  # 探索用ハイパーパラメータの設定

    params_tuning.update(params_base)
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_eval = lgb.Dataset(X_va, y_va)
  

  # 探索用ハイパーパラメータで学習
    model = lgb.train(params_tuning,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'valid'],
                    callbacks=[lgb.early_stopping(100),

                               lgb.log_evaluation(500)])

    y_va_pred = model.predict(X_va, num_iteration=model.best_iteration)
    score =  mean_absolute_error(y_va, y_va_pred)
    print('')

    return score

# ハイパーパラメータ最適化の実行

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), direction='minimize')
study.optimize(objective, n_trials=200)

# 最適化の結果確認
trial = study.best_trial
print(f'trial {trial.number}')
print('MAE bset: %.2f'% trial.value)
display(trial.params)

# 最適化ハイパーパラメータの設定
params_best = trial.params
params_best.update(params_base)
display(params_best)

#ハイパーパラメータの重要度の図示
optuna.visualization.plot_param_importances(study).show()

plt.style.use('grayscale')

# パラメータ重要度のグラフを描画
fig = optuna.visualization.matplotlib.plot_param_importances(study)
plt.show




# 最適化ハイパーパラメータを用いた学習
lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_eval = lgb.Dataset(X_va, y_va, reference=lgb_train)


# 誤差プロットの格納用データ
evals_result = {}


# 最適化ハイパーパラメータを読み込み
model = lgb.train(params_best,
                  lgb_train,
                  num_boost_round=10000,
                  valid_sets=[lgb_train, lgb_eval],
                  valid_names=['train', 'valid'],
                  callbacks=[lgb.early_stopping(100),
                             lgb.log_evaluation(500),
                             lgb.record_evaluation(evals_result)])

y_va_pred = model.predict(X_va, num_iteration=model.best_iteration)
score = mean_absolute_error(y_va, y_va_pred)
print(f'MAE valid: {score:.2f}')

# 学習データと検証データの誤差プロット
lgb.plot_metric(evals_result)


# スタイル設定
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


# グラフの作成
ax = lgb.plot_metric(evals_result)
# グラフのスタイル調整
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ラインスタイルの設定
lines = ax.get_lines()
# 学習データを点線の黒で表示
lines[0].set_color('black')
lines[0].set_linestyle('--')
# 検証データを実線の黒で表示
lines[1].set_color('black')
lines[1].set_linestyle('-')


# グリッドを非表示に
ax.grid(False)


# テキストカラーの設定
ax.tick_params(colors='black')
ax.yaxis.label.set_color('black')
ax.xaxis.label.set_color('black')
plt.title('Training History', color='black')



# 凡例のスタイル設定（修正版
handles = [lines[0], lines[1]]
labels = ['train', 'valid']
ax.legend(handles, labels, 
         facecolor='white',
         edgecolor='black',
         framealpha=1.0,
         loc='upper right')


# 凡例のテキストを黒に設定
legend = ax.get_legend()
for text in legend.get_texts():
    text.set_color('black')

# レイアウトの調整
plt.tight_layout()
plt.show()

# 検証データの予測と評価
y_va_pred = model.predict(X_va, num_iteration=model.best_iteration) 
print('MAE valid: %.2f' % (mean_absolute_error(y_va, y_va_pred)))


# テストデータの予測と評価
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration) 
print('MAE test: %.2f' % (mean_absolute_error(y_test, y_test_pred)))



# 残差のプロット

# 残差の計算
residuals = y_test - y_test_pred
# 残差と予測値の散布図
fig, ax = plt.subplots(figsize=(10, 5))  # 図と軸のオブジェクトを作成
ax.set_facecolor("white")  # 軸の背景色を白に設定  
plt.scatter(y_test_pred, residuals, s=20, color="gray")  # 散布図をプロット
# y軸の0に黒色で太く点線を追加
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

# 軸ラベルとタイトルの設定
plt.xlabel('Predicted values', fontsize=14)

plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Predicted values', fontsize=16)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

ax.tick_params(colors='black', which='both')  # ティックの色を設定
ax.spines['bottom'].set_color('black')  # X軸の色
ax.spines['left'].set_color('black')  # Y軸の色

plt.savefig('zansa.jpg', dpi=300)
plt.show()






# 特徴量重要度の可視化
importances = model.feature_importance(importance_type='gain') # 特徴量重要度
indices = np.argsort(importances)[::-1] # 特徴量重要度を降順にソート
fig, ax = plt.subplots(figsize=(20, 10))  # 図と軸のオブジェクトを作成
ax.set_facecolor("white")  # 軸の背景色を白に設定
plt.bar(range(len(indices)), importances[indices], color='gray') # 棒グラフを追加
plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
# 軸とティックの色を設定
ax.tick_params(colors='black', which='both')  # ティックの色を設定
ax.spines['bottom'].set_color('black')  # X軸の色
ax.spines['left'].set_color('black')  # Y軸の色
plt.savefig('importtant.jpg', dpi=300)
plt.show() # プロットを表示



# 特徴量重要度を取得
importances = model.feature_importance(importance_type='gain')  # 'gain' に基づく重要度
feature_names = X.columns
# データフレームの作成
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# 重要度で降順にソート
feature_importances = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# 数値として出力
print(feature_importances)



