import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from utils import load_data, calculate_rfm

st.set_page_config(page_title="Clustering", page_icon="🧩", layout="wide")
st.title("🧩 Customer Segmentation — RFM & Clustering")

df = load_data()
rfm_df = calculate_rfm(df)


# ── Step 1: RFM Feature Engineering ───────────────────────────────────────────
st.header("1️⃣ RFM Feature Engineering")


st.dataframe(rfm_df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    fig1 = px.histogram(rfm_df[rfm_df['Recency'] < rfm_df['Recency'].quantile(0.95)],
                         x='Recency', nbins=40, title='Recency Distribution',
                         color_discrete_sequence=['#e34948'])
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.histogram(rfm_df[rfm_df['Frequency'] < rfm_df['Frequency'].quantile(0.95)],
                         x='Frequency', nbins=40, title='Frequency Distribution',
                         color_discrete_sequence=['#1baf7a'])
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    fig3 = px.histogram(rfm_df[rfm_df['Monetary'] < rfm_df['Monetary'].quantile(0.95)],
                         x='Monetary', nbins=40, title='Monetary Distribution',
                         color_discrete_sequence=['#eda100'])
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Step 2: Remove Outliers + Standardize ────────────────────────────────────
st.header("2️⃣ Standardize RFM Values")

# Scale ALL customers — no outlier removal
scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
st.success("RFM values standardized using StandardScaler ✅")
st.markdown("---")
# ── Step 3 & 4: Elbow Curve ───────────────────────────────────────────────────
st.header("3️⃣ Elbow Curve — Choosing Number of Clusters")
 
inertias = []
K_range  = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)
 
fig4 = px.line(x=list(K_range), y=inertias, markers=True,
               title='Elbow Curve — Optimal Number of Clusters',
               labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'})
fig4.update_traces(line_color='#e34948')
st.plotly_chart(fig4, use_container_width=True)
 

st.markdown("---")

# ── Step 5: Run Clustering + Label ────────────────────────────────────────────
st.header("4️⃣ Run Clustering & Label Segments")


k_choice = st.slider("Select number of clusters (K)", 2, 10, 4)
# Run KMeans on clean data
km_final = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
rfm_df['Cluster'] = km_final.fit_predict(rfm_scaled).astype(str)

# Cluster averages
cluster_means = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

# Medians for labeling reference
r_med = rfm_df['Recency'].median()
f_med = rfm_df['Frequency'].median()
m_med = rfm_df['Monetary'].median()

st.write(f"📊 Median → Recency: **{r_med:.1f}** days | Frequency: **{f_med:.1f}** | Monetary: **£{m_med:.2f}**")


r_med = rfm_df['Recency'].median()
f_med = rfm_df['Frequency'].median()
m_med = rfm_df['Monetary'].median()

def label_cluster(row):

    # Rule 1 — High-Value
    # clearly recent, clearly frequent, clearly high spend
    if (row['Recency']   < r_med       and
        row['Frequency'] > f_med * 2   and   # > 4 orders
        row['Monetary']  > m_med * 2):        # > £1,337
        return 'High-Value'
    
    # Rule 3 — Occasional
    # somewhat inactive + low activity
    # Occasional: between 51 and 102 days inactive
    elif (row['Recency']   > r_med      and   # > 51 days
            row['Recency']   <= r_med * 2  and  # <= 102 days
            row['Frequency'] < f_med      and
            row['Monetary']  < m_med):
            return 'Occasional'

    # Rule 4 — At-Risk
    # very long inactive + low activity
    elif (row['Recency']   > r_med * 2 and   # > 102 days
          row['Frequency'] < f_med     and   # < 2 orders
          row['Monetary']  < m_med):          # < £668
        return 'At-Risk'

    
    # Rule 2 — Regular
    # everything else = medium F, medium M
    else:
        return 'Regular'    
# Apply labels to cluster summary
cluster_means['Segment'] = cluster_means.apply(label_cluster, axis=1)
label_map = cluster_means['Segment'].to_dict()

# Map segment to every customer
rfm_df['Segment'] = rfm_df['Cluster'].map(label_map)

# Combine clean + wholesale back together
rfm_final = pd.concat([rfm_df, rfm_df], ignore_index=True)

st.subheader("Cluster Profiles")
st.dataframe(cluster_means.round(2), use_container_width=True)

# Segment distribution
st.subheader("Segment Distribution")
seg_count = rfm_final['Segment'].value_counts().reset_index()
seg_count.columns = ['Segment', 'Count']
st.dataframe(seg_count, use_container_width=True)

st.markdown("---")

# ── Step 6: Visualize ──────────────────────────────────────────────────────────
st.header("5️⃣ Visualize Clusters")

col4, col5 = st.columns(2)
with col4:
    fig5 = px.scatter(
        rfm_final[rfm_final['Monetary'] < rfm_final['Monetary'].quantile(0.95)],
        x='Recency', y='Monetary', color='Segment', size='Frequency',
        title='Clusters — Recency vs Monetary',
        labels={'Monetary': 'Total Spend (£)'}
    )
    st.plotly_chart(fig5, use_container_width=True)
with col5:
    cluster_size = rfm_final['Segment'].value_counts().reset_index()
    cluster_size.columns = ['Segment', 'Count']
    fig6 = px.pie(cluster_size, names='Segment', values='Count',
                  title='Customer Distribution by Segment')
    st.plotly_chart(fig6, use_container_width=True)

# 3D plot
fig7 = px.scatter_3d(
    rfm_final[rfm_final['Monetary'] < rfm_final['Monetary'].quantile(0.95)],
    x='Recency', y='Frequency', z='Monetary', color='Segment',
    title='3D RFM Cluster Plot'
)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")
