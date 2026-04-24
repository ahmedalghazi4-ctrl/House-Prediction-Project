import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import os

st.set_page_config(page_title="مشروع المهندس أحمد رافد", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1e3d59;'>🏠 نظام التنبؤ الذكي ومعايير القياس العالمية</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    if not os.path.exists('final_cleaned_train.csv'):
        return None
    df = pd.read_csv('final_cleaned_train.csv')
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values
    for c in X.select_dtypes('object'):
        X[c] = X[c].astype('category').cat.codes
    
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_s = sx.fit_transform(X)
    y_s = sy.fit_transform(y.reshape(-1, 1))
    X_3d = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))

    # دالة لحساب المقاييس الثلاثة
    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100).fit(X, y)
    m_rf = get_metrics(y, rf.predict(X))

    # 2. LSTM
    lstm = Sequential([LSTM(50, activation='relu', input_shape=(1, X.shape[1])), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_3d, y_s, epochs=10, verbose=0)
    p_lstm = sy.inverse_transform(lstm.predict(X_3d))
    m_lstm = get_metrics(y, p_lstm)

    # 3. GRU
    gru = Sequential([GRU(50, activation='relu', input_shape=(1, X.shape[1])), Dense(1)])
    gru.compile(optimizer='adam', loss='mse')
    gru.fit(X_3d, y_s, epochs=10, verbose=0)
    p_gru = sy.inverse_transform(gru.predict(X_3d))
    m_gru = get_metrics(y, p_gru)
    
    return rf, lstm, gru, X.columns.tolist(), sx, sy, m_rf, m_lstm, m_gru

data_bundle = load_and_train()

if data_bundle:
    rf, lstm, gru, features, sx, sy, m_rf, m_lstm, m_gru = data_bundle
    
    st.sidebar.header("📊 مدخلات النظام")
    user_input = {f: st.sidebar.number_input(f"قيمة {f}", 0, 30000, 1000) for f in features}

    if st.sidebar.button("🚀 تحليل النتائج والمقاييس"):
        in_df = pd.DataFrame([user_input])[features]
        in_s = sx.transform(in_df)
        in_3d = in_s.reshape((1, 1, in_s.shape[1]))

        res_rf = rf.predict(in_df)[0]
        res_lstm = sy.inverse_transform(lstm.predict(in_3d))[0][0]
        res_gru = sy.inverse_transform(gru.predict(in_3d))[0][0]

        st.subheader("📋 جدول المقارنة والمعايير الإحصائية (Evaluation Metrics)")
        
        # إنشاء جدول مطابق للصورة التي أرفقتها
        results_df = pd.DataFrame({
            "Metric": ["MAE (Mean Absolute Error)", "RMSE (Root Mean Square Error)", "R² (Coefficient of Determination)"],
            "Random Forest": [f"{m_rf[0]:,.2f}", f"{m_rf[1]:,.2f}", f"{m_rf[2]:.4f}"],
            "LSTM Model": [f"{m_lstm[0]:,.2f}", f"{m_lstm[1]:,.2f}", f"{m_lstm[2]:.4f}"],
            "GRU Model": [f"{m_gru[0]:,.2f}", f"{m_gru[1]:,.2f}", f"{m_gru[2]:.4f}"]
        })
        st.table(results_df)

        st.success(f"💰 التوقع النهائي (RF): ${res_rf:,.2f} | (LSTM): ${res_lstm:,.2f} | (GRU): ${res_gru:,.2f}")
      
