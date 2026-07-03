import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_data

st.set_page_config(page_title="Recommendation", page_icon="🛒", layout="wide")
st.title("🛒 Product Recommendation System")

st.markdown("Find products frequently bought together using **Item-Based Collaborative Filtering** (Cosine Similarity).")
st.markdown("---")

df=load_data()
@st.cache_data
def build_similarity_matrix(df):
    product_matrix = (
        df.groupby(['CustomerID', 'Description'])['Quantity']
        .sum().unstack().fillna(0)
    )
    # st.dataframe(product_matrix)
    similarity = cosine_similarity(product_matrix.T)
    # st.dataframe(similarity)
    similarity_df = pd.DataFrame(
        similarity,
        index=product_matrix.columns,
        columns=product_matrix.columns
    )
    return similarity_df



with st.spinner("Building product similarity matrix... (cached after first run)"):
    similarity_df = build_similarity_matrix(df)
    # st.dataframe(similarity_df)

st.success(f"✅ Similarity matrix ready — {similarity_df.shape[0]} products indexed")
st.markdown("---")

# ── Search Box ─────────────────────────────────────────────────────────────────
st.header("🔍 Find Similar Products")

product_list = sorted(similarity_df.index.tolist())
selected_product = st.selectbox("Enter / select a product name", product_list)

top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Get Recommendations"):
    similar = (
        similarity_df[selected_product]
        .drop(selected_product)
        .sort_values(ascending=False)
        .head(top_n)
    )
    result = pd.DataFrame({
        'Product': similar.index,
        'Similarity Score': similar.values.round(4)
    })

    st.subheader(f"Top {top_n} products similar to: **{selected_product}**")
    st.dataframe(result, use_container_width=True)

    import plotly.express as px
    fig = px.bar(
        result.sort_values('Similarity Score'),
        x='Similarity Score', y='Product', orientation='h',
        title='Recommended Products', color='Similarity Score',
        color_continuous_scale='Tealgrn'
    )
    st.plotly_chart(fig, use_container_width=True)