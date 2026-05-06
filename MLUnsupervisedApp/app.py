import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Unsupervised Learning Explorer",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {color: #0C2340; font-size: 2.5rem; font-weight: bold;}
    .sub-header {color: #AE9142; font-size: 1.2rem;}
    .section-header {color: #0C2340; border-bottom: 2px solid #AE9142; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Unsupervised Learning Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">University of Notre Dame — Applied Mathematics | Vedanth Nandivada</p>', unsafe_allow_html=True)
st.markdown('---')

# Sidebar
st.sidebar.header("Configuration")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["K-Means Clustering", "Hierarchical Clustering", "PCA", "PCA + K-Means Combined"]
)

# Data loading
st.sidebar.header("Dataset")
data_source = st.sidebar.radio("Data Source", ["Sample Dataset", "Upload CSV"])

df = None
target_col = None

if data_source == "Sample Dataset":
    dataset_name = st.sidebar.selectbox("Choose Dataset", ["Iris", "Wine", "Breast Cancer"])
    if dataset_name == "Iris":
        raw = load_iris(as_frame=True)
    elif dataset_name == "Wine":
        raw = load_wine(as_frame=True)
    else:
        raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    target_col = 'target'
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        target_col = st.sidebar.selectbox("Label column (optional)", ["None"] + list(df.columns))
        if target_col == "None":
            target_col = None

if df is None:
    st.info("Please select a sample dataset or upload a CSV file to begin.")
    st.stop()

# Feature selection — exclude target by default
all_cols = list(df.columns)
default_features = [c for c in all_cols if c != target_col]
feature_cols = st.sidebar.multiselect("Features", all_cols, default=default_features)

if len(feature_cols) < 2:
    st.error("Please select at least 2 feature columns.")
    st.stop()

X_raw = df[feature_cols].dropna()
n_samples = len(X_raw)

if n_samples < 5:
    st.error(f"Dataset has only {n_samples} rows after dropping NAs. Please upload a larger dataset.")
    st.stop()

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

st.subheader(f"Dataset Preview ({n_samples} samples, {len(feature_cols)} features)")
st.dataframe(df.head(10))

# ============================================================
# K-MEANS CLUSTERING
# ============================================================
if algorithm == "K-Means Clustering":
    st.markdown('<h2 class="section-header">K-Means Clustering</h2>', unsafe_allow_html=True)

    safe_max = min(10, n_samples - 1)
    if safe_max < 2:
        st.error("Need at least 3 samples for K-Means.")
        st.stop()

    k = st.slider("Number of Clusters (k)", 2, safe_max, min(3, safe_max))
    random_state = st.slider("Random State", 0, 42, 42)

    # Elbow + Silhouette
    inertias, sil_scores = [], []
    k_range = range(2, min(safe_max + 1, n_samples))
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=random_state, n_init=10)
        labels_i = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels_i))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(list(k_range), inertias, 'bo-')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Plot')
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots()
        ax.plot(list(k_range), sil_scores, 'rs-')
        ax.set_xlabel('k')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Scores')
        st.pyplot(fig)
        plt.close()

    km_final = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km_final.fit_predict(X)
    sil = silhouette_score(X, labels)
    st.success(f"Silhouette Score for k={k}: {sil:.4f}")

    # 2D scatter via PCA
    pca2 = PCA(n_components=2)
    coords = pca2.fit_transform(X)
    plot_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    plot_df['Cluster'] = labels.astype(str)
    if target_col and target_col in df.columns:
        plot_df['True Label'] = df[target_col].values[:len(plot_df)].astype(str)
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                     title=f'K-Means Clusters (k={k}) — PCA 2D Projection',
                     hover_data=plot_df.columns.tolist())
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================
elif algorithm == "Hierarchical Clustering":
    st.markdown('<h2 class="section-header">Hierarchical Clustering</h2>', unsafe_allow_html=True)

    linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    safe_max_k = min(10, n_samples - 1)
    n_clusters = st.slider("Number of Clusters", 2, max(2, safe_max_k), min(3, safe_max_k))
    max_dendro_samples = min(100, n_samples)
    dendro_samples = st.slider("Dendrogram Sample Size", 10, max_dendro_samples,
                               min(50, max_dendro_samples))

    Z = linkage(X, method=linkage_method)
    coph_corr, _ = cophenet(Z, pdist(X))
    st.info(f"Cophenetic Correlation Coefficient: {coph_corr:.4f}")

    # Dendrogram
    fig, ax = plt.subplots(figsize=(12, 5))
    X_sub = X[:dendro_samples]
    Z_sub = linkage(X_sub, method=linkage_method)
    dendrogram(Z_sub, ax=ax, truncate_mode='lastp', p=30)
    ax.set_title(f'Dendrogram ({linkage_method} linkage, n={dendro_samples})')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Distance')
    st.pyplot(fig)
    plt.close()

    # Cut tree
    from scipy.cluster.hierarchy import fcluster
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    sil = silhouette_score(X, labels)
    st.success(f"Silhouette Score for {n_clusters} clusters: {sil:.4f}")

    # 2D scatter
    pca2 = PCA(n_components=2)
    coords = pca2.fit_transform(X)
    plot_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    plot_df['Cluster'] = labels.astype(str)
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                     title=f'Hierarchical Clusters — PCA 2D Projection')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PCA
# ============================================================
elif algorithm == "PCA":
    st.markdown('<h2 class="section-header">Principal Component Analysis (PCA)</h2>', unsafe_allow_html=True)

    max_components = min(len(feature_cols), n_samples)
    n_components = st.slider("Number of Components", 2, max_components, min(5, max_components))
    plot_type = st.radio("Scatter Plot Type", ["2D", "3D"])

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    # Scree plot
    fig, ax = plt.subplots()
    ax.bar(range(1, n_components + 1), explained * 100, color='#0C2340')
    ax.plot(range(1, n_components + 1), np.cumsum(explained) * 100, 'r-o')
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree Plot')
    st.pyplot(fig)
    plt.close()

    st.write(f"Variance explained by first 2 PCs: {sum(explained[:2])*100:.1f}%")

    color_col = None
    if target_col and target_col in df.columns:
        color_col = df[target_col].astype(str).values

    if plot_type == "2D":
        plot_df = pd.DataFrame(components[:, :2], columns=['PC1', 'PC2'])
        if color_col is not None:
            plot_df['Label'] = color_col
            fig = px.scatter(plot_df, x='PC1', y='PC2', color='Label',
                             title='2D PCA Scatter Plot')
        else:
            fig = px.scatter(plot_df, x='PC1', y='PC2', title='2D PCA Scatter Plot')
        st.plotly_chart(fig, use_container_width=True)
    else:
        if n_components < 3:
            st.warning("Need at least 3 components for 3D plot.")
        else:
            plot_df = pd.DataFrame(components[:, :3], columns=['PC1', 'PC2', 'PC3'])
            if color_col is not None:
                plot_df['Label'] = color_col
                fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3', color='Label',
                                    title='3D PCA Scatter Plot')
            else:
                fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3',
                                    title='3D PCA Scatter Plot')
            st.plotly_chart(fig, use_container_width=True)

    # Loadings heatmap
    st.subheader("Component Loadings Heatmap")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    fig, ax = plt.subplots(figsize=(min(n_components + 2, 14), max(6, len(feature_cols) // 2)))
    sns.heatmap(loadings, annot=(len(feature_cols) <= 20), fmt='.2f', cmap='coolwarm',
                center=0, ax=ax)
    ax.set_title('PCA Component Loadings')
    st.pyplot(fig)
    plt.close()

# ============================================================
# PCA + K-MEANS COMBINED
# ============================================================
elif algorithm == "PCA + K-Means Combined":
    st.markdown('<h2 class="section-header">PCA + K-Means Combined</h2>', unsafe_allow_html=True)
    st.write("Reduce dimensions with PCA first, then apply K-Means in the reduced space.")

    max_pca = min(len(feature_cols), n_samples)
    n_pca = st.slider("PCA Components", 2, max_pca, min(5, max_pca))
    safe_max_k = min(10, n_samples - 1)
    k = st.slider("Number of Clusters (k)", 2, max(2, safe_max_k), min(3, safe_max_k))
    random_state = st.slider("Random State", 0, 42, 42)

    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    st.info(f"Variance explained by {n_pca} PCs: {sum(explained)*100:.1f}%")

    # Scree
    fig, ax = plt.subplots()
    ax.bar(range(1, n_pca + 1), explained * 100, color='#0C2340')
    ax.plot(range(1, n_pca + 1), np.cumsum(explained) * 100, 'r-o')
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree Plot')
    st.pyplot(fig)
    plt.close()

    # KMeans on PCA space
    inertias, sil_scores = [], []
    k_range = range(2, min(safe_max_k + 1, n_samples))
    for ki in k_range:
        km_i = KMeans(n_clusters=ki, random_state=random_state, n_init=10)
        labs = km_i.fit_predict(X_pca)
        inertias.append(km_i.inertia_)
        sil_scores.append(silhouette_score(X_pca, labs))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(list(k_range), inertias, 'bo-')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Plot (PCA Space)')
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots()
        ax.plot(list(k_range), sil_scores, 'rs-')
        ax.set_xlabel('k')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Scores (PCA Space)')
        st.pyplot(fig)
        plt.close()

    km_final = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km_final.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    st.success(f"Silhouette Score for k={k} in PCA space: {sil:.4f}")

    # 2D scatter
    plot_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    plot_df['Cluster'] = labels.astype(str)
    if target_col and target_col in df.columns:
        plot_df['True Label'] = df[target_col].values[:len(plot_df)].astype(str)
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                     title=f'PCA + K-Means Clusters (k={k}) — PC1 vs PC2',
                     hover_data=plot_df.columns.tolist())
    st.plotly_chart(fig, use_container_width=True)

    # Loadings heatmap
    st.subheader("Component Loadings Heatmap")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f'PC{i+1}' for i in range(n_pca)]
    )
    fig, ax = plt.subplots(figsize=(min(n_pca + 2, 14), max(6, len(feature_cols) // 2)))
    sns.heatmap(loadings, annot=(len(feature_cols) <= 20), fmt='.2f', cmap='coolwarm',
                center=0, ax=ax)
    ax.set_title('PCA Component Loadings')
    st.pyplot(fig)
    plt.close()

st.markdown('---')
st.markdown('Built as part of the Notre Dame Data Science curriculum (SP26-MDSC-20009-01). | Vedanth Nandivada')
