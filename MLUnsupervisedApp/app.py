import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Unsupervised Learning Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .main-header {
        color: #0C2340;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        color: #5B6770;
        font-size: 1rem;
        margin-bottom: 0.6rem;
    }
    .section-header {
        color: #0C2340;
        border-bottom: 2px solid #AE9142;
        padding-bottom: 0.35rem;
        margin-top: 0.8rem;
    }
    .note-box {
        background: #F6F8FB;
        border-left: 4px solid #0C2340;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.7rem 0 1rem 0;
    }
    .gold-box {
        background: #FFF8E6;
        border-left: 4px solid #AE9142;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.7rem 0 1rem 0;
    }
    .small-muted {
        color: #5B6770;
        font-size: 0.92rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def note(text):
    st.markdown(f"<div class='note-box'>{text}</div>", unsafe_allow_html=True)


def gold_note(text):
    st.markdown(f"<div class='gold-box'>{text}</div>", unsafe_allow_html=True)


@st.cache_data
def load_sample_dataset(dataset_name):
    if dataset_name == "Iris":
        raw = load_iris(as_frame=True)
    elif dataset_name == "Wine":
        raw = load_wine(as_frame=True)
    else:
        raw = load_breast_cancer(as_frame=True)

    df_loaded = raw.frame.copy()
    target_names = getattr(raw, "target_names", None)
    if target_names is not None:
        df_loaded["target_name"] = [target_names[int(i)] for i in df_loaded["target"]]
    return df_loaded


def numeric_feature_options(df, target_col):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col != target_col]
