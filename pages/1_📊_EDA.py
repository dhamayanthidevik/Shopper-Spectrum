import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from utils import calculate_rfm
from utils import load_data

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum EDA",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum — EDA Dashboard")
st.markdown("---")
# ── Load Data ──────────────────────────────────────────────────────────────────
# @st.cache_data
# def load_data():
#     df = pd.read_csv('online_retail_cleaned.csv')
#     return df

df = load_data()
# sidebar for filtering

st.sidebar.header("Filter Options")
#Country filter
all_countries = ["All"]+df['Country'].unique().tolist()
selected_country = st.sidebar.selectbox("Select Country", all_countries)
# Date Range filter
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()
selected_date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date))

# Apply filters
filtered_df = df.copy()
if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]
if len(selected_date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['InvoiceDate'].dt.date >= selected_date_range[0]) &
        (filtered_df['InvoiceDate'].dt.date <= selected_date_range[1])
    ]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.subheader("📊 Key Metrics")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue",     f"£{filtered_df['TotalPrice'].sum():,.2f}")
k2.metric("Total Transactions",f"{filtered_df['InvoiceNo'].nunique():,}")
k3.metric("Total Customers",   f"{filtered_df['CustomerID'].nunique():,}")
k4.metric("Total Products",    f"{filtered_df['StockCode'].nunique():,}")

st.markdown("---")
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRANSACTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.header("1️⃣ Transaction Volume by Country")

col1, col2 = st.columns(2)

with col1:
    country_sales = (filtered_df.groupby('Country')['TotalPrice']
                     .sum().reset_index()
                     .sort_values("TotalPrice",ascending=False)
                     .head(15))
    fig=px.bar(country_sales, x='Country', y='TotalPrice', text='TotalPrice',
               labels={'TotalPrice':'Revenue (£)'}, title="Top 15 Countries by Revenue",
               color='TotalPrice', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    country_orders = (
        filtered_df.groupby('Country')['InvoiceNo']
        .nunique().reset_index()
        .sort_values('InvoiceNo', ascending=False)
        .head(15)
    )
    fig2 = px.bar(
        country_orders, x='Country', y='InvoiceNo',
        title='Top 15 Countries by Transaction Volume',
        color='InvoiceNo', color_continuous_scale='Teal',
        labels={'InvoiceNo': 'Number of Transactions'}
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOP SELLING PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════
st.header("2️⃣ Top-Selling Products")

col3, col4 = st.columns(2)

with col3:
   top_products_qty=(filtered_df.groupby('Description')['Quantity']
                                        .sum().reset_index()
                                        .sort_values(by='Quantity',ascending=False)
                                        .head(10))
   fig3=px.bar(top_products_qty, x='Description', y='Quantity', text='Quantity',
               orientation='h',
               labels={'Quantity':'Total Quantity Sold'}, title="Top 10 Products by Quantity Sold",
               color='Quantity', color_continuous_scale='Blues')
   fig3.update_layout(xaxis_tickangle=-45)
   st.plotly_chart(fig3, use_container_width=True)

with col4:
    top_products_rev = (
        filtered_df.groupby('Description')['TotalPrice']
        .sum().reset_index()
        .sort_values('TotalPrice', ascending=False)
        .head(10)
    )
    fig4 = px.bar(
        top_products_rev, x='TotalPrice', y='Description',
        orientation='h', title='Top 10 Products by Revenue',
        color='TotalPrice', color_continuous_scale='Purples',
        labels={'TotalPrice': 'Revenue (£)'}
    )
    fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PURCHASE TRENDS OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
st.header("3️⃣ Purchase Trends Over Time")

col5, col6 = st.columns(2)

with col5:
    monthly_sales = (
        filtered_df.groupby(filtered_df['InvoiceDate'].dt.to_period('M'))['TotalPrice']
        .sum().reset_index()
    )
    monthly_sales['InvoiceDate'] = monthly_sales['InvoiceDate'].astype(str)
    fig5 = px.line(
        monthly_sales, x='InvoiceDate', y='TotalPrice',
        title='Monthly Revenue Trend',
        labels={'TotalPrice': 'Revenue (£)', 'InvoiceDate': 'Month'},
        markers=True
    )
    fig5.update_traces(line_color='#2a78d6')
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    daily_orders = (
        filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['InvoiceNo']
        .nunique().reset_index()
    )
    fig6 = px.line(
        daily_orders, x='InvoiceDate', y='InvoiceNo',
        title='Daily Transaction Volume',
        labels={'InvoiceNo': 'Transactions', 'InvoiceDate': 'Date'},
        markers=False
    )
    fig6.update_traces(line_color='#1baf7a')
    st.plotly_chart(fig6, use_container_width=True)

# Day of week pattern
dow_sales = (
    filtered_df.groupby(filtered_df['InvoiceDate'].dt.day_name())['TotalPrice']
    .sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    .reset_index()
)
fig7 = px.bar(
    dow_sales, x='InvoiceDate', y='TotalPrice',
    title='Revenue by Day of Week',
    color='TotalPrice', color_continuous_scale='Viridis',
    labels={'TotalPrice': 'Revenue (£)', 'InvoiceDate': 'Day'}
)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MONETARY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.header("4️⃣ Monetary Distribution")

col7, col8 = st.columns(2)

with col7:
    fig8 = px.histogram(
        filtered_df[filtered_df['TotalPrice'] < filtered_df['TotalPrice'].quantile(0.95)],
        x='TotalPrice', nbins=50,
        title='Transaction Value Distribution (95th percentile)',
        labels={'TotalPrice': 'Transaction Value (£)'},
        color_discrete_sequence=['#2a78d6']
    )
    st.plotly_chart(fig8, use_container_width=True)
with col8:
    customer_spend = (
        filtered_df.groupby('CustomerID')['TotalPrice']
        .sum().reset_index()
    )
    fig9 = px.box(
        customer_spend[customer_spend['TotalPrice'] < customer_spend['TotalPrice'].quantile(0.95)],
        y='TotalPrice',
        title='Customer Total Spend Distribution (95th percentile)',
        labels={'TotalPrice': 'Total Spend (£)'},
        color_discrete_sequence=['#4a3aa7']
    )
    st.plotly_chart(fig9, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RFM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.header("5️⃣ RFM Analysis")


rfm = calculate_rfm(filtered_df)
col9, col10, col11 = st.columns(3)

with col9:
    fig10 = px.histogram(
        rfm[rfm['Recency'] < rfm['Recency'].quantile(0.95)],
        x='Recency', nbins=40, title='Recency Distribution',
        color_discrete_sequence=['#e34948'],
        labels={'Recency': 'Days Since Last Purchase'}
    )
    st.plotly_chart(fig10, use_container_width=True)

with col10:
    fig11 = px.histogram(
        rfm[rfm['Frequency'] < rfm['Frequency'].quantile(0.95)],
        x='Frequency', nbins=40, title='Frequency Distribution',
        color_discrete_sequence=['#1baf7a'],
        labels={'Frequency': 'Number of Purchases'}
    )
    st.plotly_chart(fig11, use_container_width=True)

with col11:
    fig12 = px.histogram(
        rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.95)],
        x='Monetary', nbins=40, title='Monetary Distribution',
        color_discrete_sequence=['#eda100'],
        labels={'Monetary': 'Total Spend (£)'}
    )
    st.plotly_chart(fig12, use_container_width=True)

# RFM Scatter
fig13 = px.scatter(
    rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.95)],
    x='Recency', y='Monetary', size='Frequency',
    title='RFM Overview — Recency vs Monetary (size = Frequency)',
    color='Frequency', color_continuous_scale='Viridis',
    labels={'Monetary': 'Total Spend (£)', 'Recency': 'Days Since Last Purchase'}
)
st.plotly_chart(fig13, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ELBOW CURVE + CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
st.header("6️⃣ Customer Segmentation — Elbow Curve & Clusters")

# Scale RFM
scaler  = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Elbow curve
inertias = []
K_range  = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

fig14 = px.line(
    x=list(K_range), y=inertias, markers=True,
    title='Elbow Curve — Optimal Number of Clusters',
    labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'}
)
fig14.update_traces(line_color='#e34948')
st.plotly_chart(fig14, use_container_width=True)


st.subheader("7️⃣ Customer Cluster Profiles")
# Choose K
k_choice = st.slider("Select number of clusters (K)", 2, 10, 4)
km_final = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
rfm['Cluster'] = km_final.fit_predict(rfm_scaled).astype(str)


cluster_profile = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
st.dataframe(cluster_profile, use_container_width=True)

col12, col13 = st.columns(2)

with col12:
    fig15 = px.scatter(
        rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.95)],
        x='Recency', y='Monetary', color='Cluster',
        title='Clusters — Recency vs Monetary',
        labels={'Monetary': 'Total Spend (£)'},
        size='Frequency'
    )
    st.plotly_chart(fig15, use_container_width=True)

with col13:
    cluster_size = rfm['Cluster'].value_counts().reset_index()
    cluster_size.columns = ['Cluster', 'Count']
    fig16 = px.pie(
        cluster_size, names='Cluster', values='Count',
        title='Customer Distribution by Cluster'
    )
    st.plotly_chart(fig16, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PRODUCT RECOMMENDATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
st.header("8️⃣ Product Similarity Matrix")

st.info("Showing top 15 products by revenue for similarity analysis.")

top15 = (
    filtered_df.groupby('Description')['TotalPrice']
    .sum().nlargest(15).index.tolist()
)

basket = (
    filtered_df[filtered_df['Description'].isin(top15)]
    .groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum().unstack().fillna(0)
)
basket = (basket > 0).astype(int)

similarity = basket.corr(method='pearson').round(2)

fig17 = px.imshow(
    similarity,
    title='Product Co-purchase Similarity Matrix (Top 15 Products)',
    color_continuous_scale='RdBu',
    zmin=-1, zmax=1,
    aspect='auto'
)
fig17.update_layout(height=600)
st.plotly_chart(fig17, use_container_width=True)

st.markdown("---")
st.caption("Shopper Spectrum EDA Dashboard — Built with Streamlit & Plotly")
