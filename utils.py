import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_csv('online_retail_cleaned.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


@st.cache_data
def calculate_rfm(df):
    """Calculate Recency, Frequency, Monetary for each customer."""
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('InvoiceNo',   'nunique'),
        Monetary  = ('TotalPrice',  'sum')
    ).reset_index()

    return rfm