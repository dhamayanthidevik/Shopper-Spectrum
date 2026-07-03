import streamlit as st

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.markdown("### Customer Segmentation & Product Recommendation System")

st.markdown("---")

st.markdown("""
Welcome! Use the **sidebar** to navigate between pages:

- **📊 EDA** — Explore transaction trends, top products, sales by country
- **🧩 Clustering** — RFM analysis, elbow curve, customer segments
- **🛒 Recommendation** — Get similar product suggestions

👈 Select a page from the sidebar to get started.
""")

st.markdown("---")
st.caption("Shopper Spectrum — Built with Streamlit")