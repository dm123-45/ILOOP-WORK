# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ILOOP — Data Analysis & Visualization", layout="wide")

# --- Header
st.title("ILOOP — Data Analysis & Visualization")
st.write(
    "Upload a CSV from the ILOOP project (or any CSV). "
    "The app provides quick EDA (head, summary, missing values) and interactive visualisations."
)

# --- Sidebar controls
st.sidebar.header("Upload / Options")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv", "txt"])
use_sample = st.sidebar.checkbox("Use sample dataset (Iris)", value=False)
st.sidebar.markdown("---")
st.sidebar.header("Plot options")
plot_type = st.sidebar.selectbox("Plot type", ["Histogram", "Boxplot", "Scatter", "Correlation heatmap"])
selected_cols = st.sidebar.multiselect("Columns (for plots)", [], help="Choose columns for plotting (will update once data loaded)")

# --- Load data
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

df = None
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.success("CSV loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
elif use_sample:
    df = px.data.iris()
    st.info("Loaded sample Iris dataset.")

if df is None:
    st.warning("Please upload a CSV or choose the sample dataset to begin.")
    st.stop()

# --- Show basic info
with st.expander("Dataset preview & basic info", expanded=True):
    st.subheader("Preview")
    st.dataframe(df.head(100))

    st.subheader("Basic info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

# --- Summary statistics & missing values
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Summary statistics")
    st.dataframe(df.describe(include='all').transpose())

with col2:
    st.subheader("Missing values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.write("No missing values detected.")
    else:
        st.table(missing)

# --- Column selection helper
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()
if not selected_cols:
    # sensible defaults
    selected_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else all_cols[:2]

# Update sidebar chosen list for clarity (will not re-render selection widget)
st.sidebar.write("Detected numeric columns:", numeric_cols)

# --- Plots
st.subheader("Visualisations")

if plot_type == "Histogram":
    col = st.selectbox("Choose numeric column", numeric_cols, index=0 if numeric_cols else 0)
    bins = st.slider("Bins", min_value=5, max_value=200, value=30)
    fig = px.histogram(df, x=col, nbins=bins, marginal="rug", title=f"Histogram of {col}")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Boxplot":
    col = st.selectbox("Choose numeric column", numeric_cols, index=0 if numeric_cols else 0)
    fig = px.box(df, y=col, points="all", title=f"Boxplot of {col}")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Scatter":
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns for scatter plot.")
    else:
        x_col = st.selectbox("X axis", numeric_cols, index=0)
        y_col = st.selectbox("Y axis", numeric_cols, index=1)
        color_col = st.selectbox("Color (optional)", [None] + all_cols, index=0)
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, trendline="ols", title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Correlation heatmap":
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns to compute correlations.")
    else:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix (numeric columns)")
        st.plotly_chart(fig, use_container_width=True)

# --- Pairwise / advanced view
st.subheader("Advanced: Pairplot / Distribution matrix")
if st.checkbox("Show pairplot (may be slow for large datasets)"):
    pair_cols = st.multiselect("Choose columns for pairplot (numeric)", numeric_cols, default=numeric_cols[:4])
    if len(pair_cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=pair_cols, title="Scatter matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select two or more numeric columns.")

# --- Filtering & download cleaned data
st.subheader("Filter & export cleaned dataset")
filter_col = st.selectbox("Column to filter on (optional)", [None] + all_cols)
filtered_df = df.copy()
if filter_col:
    unique_vals = df[filter_col].dropna().unique().tolist()
    sel = st.multiselect(f"Select values from {filter_col}", unique_vals, default=unique_vals[:1])
    if sel:
        filtered_df = df[df[filter_col].isin(sel)]
        st.write(f"Filtered shape: {filtered_df.shape}")
    else:
        st.write("No values selected — no filtering applied.")

st.download_button(
    label="Download filtered dataset as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_dataset.csv",
    mime="text/csv",
)

# --- Notes & next steps
st.markdown("---")
st.subheader("Notes")
st.write(
    "• This app is a generic EDA & visualization scaffold tailored for the ILOOP data analysis + visualization workflow. "
    "Replace or extend plotting sections with the specific plots or model outputs used in your notebook from ILOOP.pdf. "
)
st.write("• If you want, I can convert specific code cells from your `ILOOP.pdf` into Streamlit widgets/plots — tell me which sections to extract.")


