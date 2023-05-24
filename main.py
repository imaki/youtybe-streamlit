import streamlit as st
import numpy as np
import pandas as pd

# タイトルを表示する
st.title("タイトルの表示")

# サンプルのデータフレームを作成する
df = pd.DataFrame({'A': [1, 2, 3],
                   'B': [4, 5, 6],
                   'C': [7, 8, 9]})

# 追加する四列のデータを作成する
col_d = [10, 11, 12]
col_e = [13, 14, 15]
col_f = [16, 17, 18]
col_g = [19, 20, 21]

# データフレームに四列を追加する
# df = df.assign(D=col_d, E=col_e, F=col_f, G=col_g)

# データフレームを表示する
st.write(df)
