import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.title('Segmentasi Pelanggan Online Retail')
data = pd.read_csv('segmentasi_pelanggan.csv')
scaler = joblib.load('rfm_scaler.joblib')
model = joblib.load('rfm_kmeans.joblib')

col1, col2 = st.columns(2)
col1.metric('Jumlah pelanggan', len(data))
col2.metric('Jumlah cluster', data['Cluster'].nunique())
st.subheader('Distribusi pelanggan')
st.bar_chart(data['Segment'].value_counts())
st.subheader('Profil RFM per cluster')
st.dataframe(data.groupby(['Cluster', 'Segment'])[['Recency', 'Frequency', 'Monetary']].mean().round(2))
st.subheader('Daftar pelanggan')
st.dataframe(data)

st.subheader('Scatterplot K-Means (RFM)')
st.caption('Setiap titik = 1 pelanggan. Warna = cluster; sumbu log agar nilai ekstrem tetap terbaca.')
axes = [('Recency', 'Frequency'), ('Recency', 'Monetary'), ('Frequency', 'Monetary')]
for left, right in axes:
    chart = alt.Chart(data).mark_circle(size=45, opacity=0.55).encode(
        x=alt.X(f'{left}:Q', scale=alt.Scale(type='log'), title=left),
        y=alt.Y(f'{right}:Q', scale=alt.Scale(type='log'), title=right),
        color=alt.Color('Segment:N', title='Cluster / segmen'),
        tooltip=['CustomerID:O', 'Cluster:N', 'Segment:N', 'Recency:Q', 'Frequency:Q', 'Monetary:Q'],
    ).properties(height=340)
    st.altair_chart(chart, width='stretch')
st.caption('K-Means dilatih pada log1p(RFM) yang distandardisasi; grafik ini memperlihatkan nilai RFM asli, bukan bidang keputusan model.')

st.subheader('Prediksi segmen pelanggan baru')
recency = st.number_input('Recency (hari)', min_value=0, value=30)
frequency = st.number_input('Frequency (jumlah invoice)', min_value=1, value=2)
monetary = st.number_input('Monetary (total belanja)', min_value=0.0, value=500.0)
if st.button('Prediksi'):
    fitur = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    cluster = int(model.predict(scaler.transform(np.log1p(fitur)))[0])
    segment = data.loc[data['Cluster'] == cluster, 'Segment'].iloc[0]
    st.success(f'Cluster {cluster} — {segment}')
