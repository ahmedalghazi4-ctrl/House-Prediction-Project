import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

st.set_page_config(page_title="مشروع المهندس أحمد رافد", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1e3d59;'>🏠 نظام التنبؤ الذكي والمقارنة (النسخة السريعة)</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    if not os.path.exists('final_cleaned_train.csv'): return None
    df = pd.read_csv('final_cleaned_train.csv')
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values
    for c in X.select_dtypes('object'): X[c] = X[c].astype('category').cat.codes
    
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_s = sx.fit_transform(X)
    y_s = sy.fit_transform(y.reshape(-1, 1)).ravel()

    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100).fit(X, y)
    m_rf = get_metrics(y, rf.predict(X))

    # 2. Artificial Neural Network (ANN)
    ann = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500).fit(X_s, y_s)
    p_ann = sy.inverse_transform(ann.predict(X_s).reshape(-1, 1))
    m_ann = get_metrics(y, p_ann)

    return rf, ann, X.columns.tolist(), sx, sy, m_rf, m_ann

data_bundle = load_and_train()
if data_bundle:
    rf, ann, features, sx, sy, m_rf, m_ann = data_bundle
    st.sidebar.header("📊 أدخل مواصفات العقار")
    user_input = {f: st.sidebar.number_input(f"قيمة {f}", 0.0, 1000000.0, 1000.0) for f in features}

    if st.sidebar.button("🚀 تشغيل التحليل والمقارنة"):
        in_df = pd.DataFrame([user_input])[features]
        in_s = sx.transform(in_df)
        
        res_rf = rf.predict(in_df)[0]
        res_ann = sy.inverse_transform(ann.predict(in_s).reshape(-1, 1))[0][0]

        st.subheader("🏁 نتائج التقييم (Evaluation Metrics)")
        results_df = pd.DataFrame({
            "المعيار (Metric)": ["MAE", "RMSE", "R² Score"],
            "Random Forest": [f"{m_rf[0]:,.2f}", f"{m_rf[1]:,.2f}", f"{m_rf[2]:.4f}"],
            "Neural Network (ANN)": [f"{m_ann[0]:,.2f}", f"{m_ann[1]:,.2f}", f"{m_ann[2]:.4f}"]
        })
        st.table(results_df)
        
        st.success(f"💰 التوقع النهائي: (RF) ${res_rf:,.2f} | (ANN) ${res_ann:,.2f}")
        
