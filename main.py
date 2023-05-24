import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# フォントの設定（日本語フォントのパスを指定してください）
font_path = 'C:/Windows/Fonts/MSGOTHIC.TTC'
fontprop = fm.FontProperties(fname=font_path)

# 株価データの読み込み
df = pd.read_csv('C:/Users/not/Documents/AMZN 過去データ.csv')

# データの前処理
window = 5
df['SMA'] = df['終値'].rolling(window=window).mean()
df.dropna(inplace=True)

# 特徴量と目的変数の分割
X = df[['SMA']]
y = df['終値']

# トレーニングデータとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルのトレーニング
model = LinearRegression()
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# グラフの作成
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['終値'], label='実測値', linewidth=2)
plt.plot(X_test.index, y_pred, label='予測値', linewidth=2)
plt.plot(df.index, df['SMA'], label='単純移動平均線', linewidth=2)
plt.xlabel('日付', fontproperties=fontprop)
plt.ylabel('終値', fontproperties=fontprop)
plt.title('株価予測 - 単純移動平均線', fontproperties=fontprop)
plt.legend(prop=fontprop)

# グラフの表示
plt.show()
