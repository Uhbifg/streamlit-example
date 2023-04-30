from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
import numpy as np

st.title("Предсказание времени простоя поезда!")
X_train = pd.read_parquet("data/train.parquet")
uploaded_file = st.file_uploader("Choose a file (parquet-like)")
if uploaded_file is not None:
  X_test = pd.read_parquet(uploaded_file)
  st.write(df)

    
    
from_file = CatBoostRegressor()
from_file.load_model("model_last")


@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

test_res = np.exp(model.predict(X_test.drop(columns=shit_cols)))
answer = pd.DataFrame()
answer["target"] = test_res

csv = convert_df(answer)

st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)
