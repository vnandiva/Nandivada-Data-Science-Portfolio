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


def remove_constant_columns(df, feature_cols):
    constant_cols = [col for col in feature_cols if df[col].nunique(dropna=True) <= 1]
    usable_cols = [col for col in feature_cols if col not in constant_cols]
    return usable_cols, constant_cols


def preprocess_features(df, feature_cols):
    X_raw = df.loc[:, feature_cols]
    missing_count = int(X_raw.isna().sum().sum())
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, missing_count


def safe_silhouette(X, labels):
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return None
    return silhouette_score(X, labels)


def score_label(score):
    if score is None:
        return "not available"
    if score >= 0.7:
        return "very strong"
    if score >= 0.5:
        return "strong"
    if score >= 0.25:
        return "moderate"
    if score >= 0:
        return "weak"
    return "poor"


def plot_silhouette(X, labels, title):
    n_clusters = len(np.unique(labels))
    sil_vals = silhouette_samples(X, labels)
    avg_score = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y_lower = 10
    cmap = cm.get_cmap("tab10")
    for cluster_id in range(n_clusters):
        cluster_vals = np.sort(sil_vals[labels == cluster_id])
        cluster_size = cluster_vals.shape[0]
        y_upper = y_lower + cluster_size
        color = cmap(cluster_id / max(n_clusters, 1))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
        )
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(cluster_id + 1), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(avg_score, color="red", linestyle="--", label=f"Average = {avg_score:.3f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def pca_dataframe(X, labels=None, target_values=None, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    plot_df = pd.DataFrame(coords, columns=columns)
    if labels is not None:
        plot_df["Cluster"] = [f"Cluster {int(i) + 1}" for i in labels]
    if target_values is not None:
        plot_df["Reference label"] = target_values.astype(str).values
    return pca, plot_df


def render_metric_interpretation(silhouette_value=None, davies_value=None, context=""):
    parts = []
    if silhouette_value is not None:
        parts.append(
            f"The silhouette score is <b>{silhouette_value:.3f}</b>, which is "
            f"<b>{score_label(silhouette_value)}</b>. A score near 1 means points fit their "
            "own cluster well; a score near 0 means clusters overlap; a negative score suggests "
            "some points may be assigned to the wrong cluster."
        )
    if davies_value is not None:
        parts.append(
            f"The Davies-Bouldin index is <b>{davies_value:.3f}</b>. Lower values are better "
            "because they mean clusters are compact and well separated."
        )
    if context:
        parts.append(context)
    if parts:
        gold_note(" ".join(parts))


def render_scree_plot(explained, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    component_numbers = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained) * 100
    ax.bar(component_numbers, explained * 100, color="#0C2340", alpha=0.85, label="Individual")
    ax.plot(component_numbers, cumulative, "o-", color="#AE9142", linewidth=2.5, label="Cumulative")
    ax.axhline(80, color="gray", linestyle=":", linewidth=1.2, label="80% reference")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def sidebar_int_slider(label, min_value, max_value, value, step=1, help_text=None):
    if max_value <= min_value:
        st.sidebar.caption(f"{label}: fixed at {min_value}")
        return min_value
    return st.sidebar.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=min(value, max_value),
        step=step,
        help=help_text,
    )


st.markdown("<p class='main-header'>Unsupervised Learning Explorer</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>University of Notre Dame | Applied Mathematics | Vedanth Nandivada</p>",
    unsafe_allow_html=True,
)

with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
        1. Choose a built-in dataset or upload a CSV file.
        2. Select numeric feature columns. These are the measurements the model uses to find patterns.
        3. Choose an unsupervised learning method and tune its settings.
        4. Use the plots and scores to decide whether the discovered structure looks meaningful.

        **Beginner note:** Unsupervised learning means the model is not given the correct answer.
        It looks for hidden structure on its own. In this app, that usually means grouping similar rows
        into clusters or compressing many columns into a few important directions with PCA.
        """
    )

st.markdown("---")


st.sidebar.header("1. Choose Data")
data_source = st.sidebar.radio("Data source", ["Sample Dataset", "Upload CSV"])

df = None
target_col = None

if data_source == "Sample Dataset":
    dataset_name = st.sidebar.selectbox("Sample dataset", ["Iris", "Wine", "Breast Cancer"])
    df = load_sample_dataset(dataset_name)
    target_col = "target"
    st.sidebar.caption("The target column is kept only as a reference label, not as a modeling feature.")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if df.empty:
                st.error("The uploaded CSV is empty. Please upload a file with rows and columns.")
                st.stop()
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()

if df is None:
    st.info("Choose a sample dataset or upload a CSV file to begin.")
    st.stop()

if data_source == "Upload CSV":
    label_options = ["None"] + df.columns.tolist()
    target_choice = st.sidebar.selectbox(
        "Optional label/color column",
        label_options,
        help="This column is not used by the model. It is only used as a reference in plots.",
    )
    target_col = None if target_choice == "None" else target_choice

feature_options = numeric_feature_options(df, target_col)
if len(feature_options) < 2:
    st.error("This dataset needs at least two numeric feature columns for clustering or PCA.")
    st.stop()

default_features = feature_options[: min(6, len(feature_options))]

st.sidebar.header("2. Select Features")
feature_cols = st.sidebar.multiselect(
    "Numeric model features",
    feature_options,
    default=default_features,
    help="Only numeric columns are shown here because scaling and ML models require numeric input.",
)

if len(feature_cols) < 2:
    st.error("Please select at least two numeric feature columns.")
    st.stop()

feature_cols, constant_cols = remove_constant_columns(df, feature_cols)
if constant_cols:
    st.warning(
        "These selected columns had no variation and were removed because they cannot help the model: "
        + ", ".join(constant_cols)
    )

if len(feature_cols) < 2:
    st.error("After removing constant columns, fewer than two usable features remain.")
    st.stop()

n_samples = len(df)
if n_samples < 5:
    st.error(f"This dataset has only {n_samples} rows. Please use at least 5 rows.")
    st.stop()

X, missing_count = preprocess_features(df, feature_cols)

st.sidebar.header("3. Choose Method")
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["K-Means Clustering", "Hierarchical Clustering", "PCA", "PCA + K-Means Combined"],
)

st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Selected features", len(feature_cols))
col3.metric("Missing values imputed", missing_count)
col4.metric("Reference label", target_col if target_col else "None")

with st.expander("Preview selected data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("**Selected numeric features:** " + ", ".join(feature_cols))

note(
    "Before modeling, selected numeric features are median-imputed if values are missing, then standardized. "
    "Scaling matters because clustering depends on distances; without scaling, a column with large numbers can "
    "dominate the result even if it is not more important."
)


if algorithm == "K-Means Clustering":
    st.markdown("<h2 class='section-header'>K-Means Clustering</h2>", unsafe_allow_html=True)
    note(
        "<b>What K-Means does:</b> K-Means tries to divide rows into <i>k</i> groups. "
        "It starts with tentative cluster centers, assigns each row to the nearest center, then moves the centers "
        "until the assignments stabilize. Use it when you want compact groups of similar rows."
    )

    st.sidebar.header("4. Tune K-Means")
    max_k = min(10, n_samples - 1)
    k = sidebar_int_slider("Number of clusters (k)", 2, max_k, min(3, max_k))
    init_method = st.sidebar.selectbox("Initialization method", ["k-means++", "random"])
    n_init = sidebar_int_slider("Number of restarts (n_init)", 5, 30, 10)
    max_iter = sidebar_int_slider("Maximum iterations", 100, 1000, 300, step=50)
    random_state = sidebar_int_slider("Random seed", 0, 100, 42)

    gold_note(
        "<b>How to tune:</b> <i>k</i> is the number of groups you ask the model to find. "
        "<i>n_init</i> reruns K-Means from different starting points and keeps the best result. "
        "<i>max_iter</i> limits how long each run can refine its centers."
    )

    k_range = range(2, max_k + 1)
    inertias, sil_scores = [], []
    for candidate_k in k_range:
        km = KMeans(
            n_clusters=candidate_k,
            init=init_method,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels_candidate = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(safe_silhouette(X, labels_candidate))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(k_range), inertias, "o-", color="#0C2340")
        ax.axvline(k, color="#AE9142", linestyle="--", label=f"Selected k={k}")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Plot", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(k_range), sil_scores, "s-", color="#AE9142")
        ax.axvline(k, color="#0C2340", linestyle="--", label=f"Selected k={k}")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette by k", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    gold_note(
        "<b>How to read these:</b> In the elbow plot, look for the point where inertia stops dropping quickly. "
        "In the silhouette plot, higher is better. Use both charts together rather than trusting one number blindly."
    )

    km_final = KMeans(
        n_clusters=k,
        init=init_method,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = km_final.fit_predict(X)
    sil = safe_silhouette(X, labels)
    db = davies_bouldin_score(X, labels)

    m1, m2, m3 = st.columns(3)
    m1.metric("Inertia", f"{km_final.inertia_:,.2f}")
    m2.metric("Silhouette", f"{sil:.3f}" if sil is not None else "N/A")
    m3.metric("Davies-Bouldin", f"{db:.3f}")
    render_metric_interpretation(
        sil,
        db,
        "If the score is weak, try a different k or remove features that do not seem relevant.",
    )

    pca2, plot_df = pca_dataframe(
        X,
        labels=labels,
        target_values=df[target_col] if target_col in df.columns else None,
        n_components=2,
    )
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        symbol="Reference label" if "Reference label" in plot_df.columns else None,
        title=f"K-Means Clusters (k={k}) in a 2D PCA Projection",
        hover_data=plot_df.columns.tolist(),
    )
    st.plotly_chart(fig, use_container_width=True)
    note(
        "This plot uses PCA only for visualization. K-Means was trained on the full selected feature set, "
        "then the rows were projected into two dimensions so you can inspect the cluster separation."
    )

    fig = plot_silhouette(X, labels, f"Silhouette Plot for K-Means (k={k})")
    st.pyplot(fig)
    plt.close(fig)


elif algorithm == "Hierarchical Clustering":
    st.markdown("<h2 class='section-header'>Hierarchical Clustering</h2>", unsafe_allow_html=True)
    note(
        "<b>What hierarchical clustering does:</b> It starts with every row as its own cluster, then repeatedly "
        "merges the closest clusters. The dendrogram shows that merging history as a tree."
    )

    if n_samples > 1200:
        st.warning(
            "Hierarchical clustering can be slow on large datasets because it compares many pairs of rows. "
            "For best performance, use fewer rows or fewer selected features."
        )

    st.sidebar.header("4. Tune Hierarchical Clustering")
    max_k = min(10, n_samples - 1)
    n_clusters = sidebar_int_slider("Number of clusters", 2, max_k, min(3, max_k))
    linkage_method = st.sidebar.selectbox("Linkage method", ["ward", "complete", "average", "single"])
    if linkage_method == "ward":
        distance_metric = "euclidean"
        st.sidebar.caption("Ward linkage requires Euclidean distance.")
    else:
        distance_metric = st.sidebar.selectbox("Distance metric", ["euclidean", "cityblock", "cosine"])

    max_dendro_samples = min(100, n_samples)
    min_dendro_samples = min(10, max_dendro_samples)
    dendro_samples = sidebar_int_slider(
        "Dendrogram sample size",
        min_dendro_samples,
        max_dendro_samples,
        min(50, max_dendro_samples),
    )

    gold_note(
        "<b>How to tune:</b> Linkage controls how the distance between two clusters is calculated. "
        "Ward often makes compact clusters; complete is stricter; average is balanced; single can create long chains."
    )

    Z = linkage(X, method=linkage_method, metric=distance_metric)
    raw_labels = fcluster(Z, n_clusters, criterion="maxclust")
    unique_raw = {label: idx for idx, label in enumerate(sorted(np.unique(raw_labels)))}
    labels = np.array([unique_raw[label] for label in raw_labels])
    sil = safe_silhouette(X, labels)
    db = davies_bouldin_score(X, labels)

    coph_corr, _ = cophenet(Z, pdist(X, metric=distance_metric))

    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette", f"{sil:.3f}" if sil is not None else "N/A")
    m2.metric("Davies-Bouldin", f"{db:.3f}")
    m3.metric("Cophenetic correlation", f"{coph_corr:.3f}")
    render_metric_interpretation(
        sil,
        db,
        "Cophenetic correlation measures how faithfully the dendrogram preserves the original pairwise distances. "
        "Values closer to 1 mean the tree is a better summary of the data geometry.",
    )

    X_dendro = X[:dendro_samples]
    Z_dendro = linkage(X_dendro, method=linkage_method, metric=distance_metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z_dendro, ax=ax, truncate_mode="lastp", p=min(30, dendro_samples))
    ax.set_title(f"Dendrogram ({linkage_method} linkage, {dendro_samples} displayed rows)", fontweight="bold")
    ax.set_xlabel("Rows or merged groups")
    ax.set_ylabel("Distance")
    st.pyplot(fig)
    plt.close(fig)

    note(
        "In a dendrogram, lower merges mean rows or groups are more similar. Larger vertical jumps suggest a natural "
        "place to separate clusters."
    )

    _, plot_df = pca_dataframe(
        X,
        labels=labels,
        target_values=df[target_col] if target_col in df.columns else None,
        n_components=2,
    )
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        symbol="Reference label" if "Reference label" in plot_df.columns else None,
        title="Hierarchical Clusters in a 2D PCA Projection",
        hover_data=plot_df.columns.tolist(),
    )
    st.plotly_chart(fig, use_container_width=True)

    if sil is not None:
        fig = plot_silhouette(X, labels, f"Silhouette Plot for Hierarchical Clustering ({n_clusters} clusters)")
        st.pyplot(fig)
        plt.close(fig)


elif algorithm == "PCA":
    st.markdown("<h2 class='section-header'>Principal Component Analysis (PCA)</h2>", unsafe_allow_html=True)
    note(
        "<b>What PCA does:</b> PCA turns several related feature columns into new columns called principal components. "
        "The first component captures the strongest direction of variation, the second captures the next strongest, "
        "and so on. PCA is useful for visualization and compression."
    )

    st.sidebar.header("4. Tune PCA")
    max_components = min(len(feature_cols), n_samples)
    n_components = sidebar_int_slider("Number of components", 2, max_components, min(5, max_components))
    plot_type = st.sidebar.radio("Scatter plot type", ["2D", "3D"])

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    render_scree_plot(explained, "PCA Scree Plot")
    retained = float(np.sum(explained) * 100)
    st.info(f"The selected {n_components} components retain {retained:.1f}% of the variance in the selected features.")
    gold_note(
        "<b>Beginner interpretation:</b> Variance explained is the amount of information kept by each component. "
        "If the first few components explain a high percentage, the dataset can be summarized well in fewer dimensions."
    )

    target_values = df[target_col] if target_col in df.columns else None
    if plot_type == "2D":
        plot_df = pd.DataFrame(components[:, :2], columns=["PC1", "PC2"])
        if target_values is not None:
            plot_df["Reference label"] = target_values.astype(str).values
        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Reference label" if "Reference label" in plot_df.columns else None,
            title="PCA Scatter: PC1 vs PC2",
            hover_data=plot_df.columns.tolist(),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        if n_components < 3:
            st.warning("Choose at least 3 PCA components to display a 3D scatterplot.")
        else:
            plot_df = pd.DataFrame(components[:, :3], columns=["PC1", "PC2", "PC3"])
            if target_values is not None:
                plot_df["Reference label"] = target_values.astype(str).values
            fig = px.scatter_3d(
                plot_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Reference label" if "Reference label" in plot_df.columns else None,
                title="PCA Scatter: PC1 vs PC2 vs PC3",
                hover_data=plot_df.columns.tolist(),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Component Loadings Heatmap")
    note(
        "Loadings show which original features contribute most to each principal component. Large positive or negative "
        "values mean the feature strongly influences that component."
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    fig, ax = plt.subplots(figsize=(min(n_components + 3, 14), max(5, len(feature_cols) * 0.35)))
    sns.heatmap(loadings, annot=len(feature_cols) <= 20, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("PCA Component Loadings", fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)


elif algorithm == "PCA + K-Means Combined":
    st.markdown("<h2 class='section-header'>PCA + K-Means Combined</h2>", unsafe_allow_html=True)
    note(
        "<b>Why combine them:</b> PCA first compresses the selected features into fewer components. "
        "K-Means then clusters the compressed data. This can help when many features are noisy or redundant, "
        "but keeping too few components can remove useful information."
    )

    st.sidebar.header("4. Tune PCA + K-Means")
    max_pca = min(len(feature_cols), n_samples)
    n_pca = sidebar_int_slider("PCA components", 2, max_pca, min(5, max_pca))
    max_k = min(10, n_samples - 1)
    k = sidebar_int_slider("Number of clusters (k)", 2, max_k, min(3, max_k))
    init_method = st.sidebar.selectbox("Initialization method", ["k-means++", "random"])
    n_init = sidebar_int_slider("Number of restarts (n_init)", 5, 30, 10)
    max_iter = sidebar_int_slider("Maximum iterations", 100, 1000, 300, step=50)
    random_state = sidebar_int_slider("Random seed", 0, 100, 42)

    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    render_scree_plot(explained, "PCA Scree Plot Before K-Means")
    st.info(f"The PCA step keeps {np.sum(explained) * 100:.1f}% of the original variance.")

    k_range = range(2, max_k + 1)
    inertias, sil_scores = [], []
    for candidate_k in k_range:
        km = KMeans(
            n_clusters=candidate_k,
            init=init_method,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels_candidate = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil_scores.append(safe_silhouette(X_pca, labels_candidate))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(k_range), inertias, "o-", color="#0C2340")
        ax.axvline(k, color="#AE9142", linestyle="--", label=f"Selected k={k}")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Plot in PCA Space", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(k_range), sil_scores, "s-", color="#AE9142")
        ax.axvline(k, color="#0C2340", linestyle="--", label=f"Selected k={k}")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette by k in PCA Space", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    km_pca = KMeans(
        n_clusters=k,
        init=init_method,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels_pca = km_pca.fit_predict(X_pca)
    sil_pca = safe_silhouette(X_pca, labels_pca)
    db_pca = davies_bouldin_score(X_pca, labels_pca)

    km_original = KMeans(
        n_clusters=k,
        init=init_method,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels_original = km_original.fit_predict(X)
    sil_original = safe_silhouette(X, labels_original)

    m1, m2, m3 = st.columns(3)
    m1.metric("PCA-space silhouette", f"{sil_pca:.3f}" if sil_pca is not None else "N/A")
    m2.metric("Original-space silhouette", f"{sil_original:.3f}" if sil_original is not None else "N/A")
    m3.metric("PCA-space Davies-Bouldin", f"{db_pca:.3f}")
    render_metric_interpretation(
        sil_pca,
        db_pca,
        "Compare the PCA-space silhouette to the original-space silhouette. If PCA-space is higher, compression may be helping the cluster structure stand out.",
    )

    plot_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    plot_df["Cluster"] = [f"Cluster {int(i) + 1}" for i in labels_pca]
    if target_col in df.columns:
        plot_df["Reference label"] = df[target_col].astype(str).values
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        symbol="Reference label" if "Reference label" in plot_df.columns else None,
        title=f"PCA + K-Means Clusters (k={k})",
        hover_data=plot_df.columns.tolist(),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Component Loadings Heatmap")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i + 1}" for i in range(n_pca)],
    )
    fig, ax = plt.subplots(figsize=(min(n_pca + 3, 14), max(5, len(feature_cols) * 0.35)))
    sns.heatmap(loadings, annot=len(feature_cols) <= 20, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("PCA Component Loadings", fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)


st.markdown("---")
st.markdown(
    "<p class='small-muted'>Built as part of the Notre Dame Data Science curriculum "
    "(SP26-MDSC-20009-01). | Vedanth Nandivada</p>",
    unsafe_allow_html=True,
)
