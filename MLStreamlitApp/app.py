import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Explorer", page_icon="🤖", layout="wide")

st.title("🤖 Interactive Machine Learning Explorer")
st.write(
    "Upload a CSV dataset (or use the built-in sample), select a supervised ML model, "
    "tune hyperparameters, and explore model performance — all without writing a single line of code."
)

# --- Sample datasets ---
@st.cache_data
def load_sample_iris():
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

@st.cache_data
def load_sample_wine():
    from sklearn.datasets import load_wine
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

@st.cache_data
def load_sample_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

st.divider()

# --- Sidebar: Data ---
st.sidebar.header("1. Data")
data_source = st.sidebar.radio(
    "Data source",
    ["Sample: Iris", "Sample: Wine", "Sample: Breast Cancer", "Upload CSV"]
)

df = None
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
elif data_source == "Sample: Iris":
    df = load_sample_iris()
elif data_source == "Sample: Wine":
    df = load_sample_wine()
else:
    df = load_sample_breast_cancer()

if df is None:
    st.info("Please select a sample dataset or upload a CSV file to get started.")
    st.stop()

st.subheader("Dataset Preview")
st.caption(f"Loaded **{len(df):,}** rows x **{df.shape[1]}** columns.")
st.dataframe(df.head(10), use_container_width=True, height=260)

st.divider()

# --- Sidebar: Feature / Target Selection ---
st.sidebar.header("2. Features & Target")
all_cols = df.columns.tolist()
target_col = st.sidebar.selectbox("Target column", all_cols, index=len(all_cols)-1)
feature_cols = st.sidebar.multiselect(
    "Feature columns",
    [c for c in all_cols if c != target_col],
    default=[c for c in all_cols if c != target_col]
)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# Encode target if needed
df_model = df[feature_cols + [target_col]].dropna()
X = df_model[feature_cols]
le = LabelEncoder()
y = le.fit_transform(df_model[target_col].astype(str))
n_classes = len(np.unique(y))

# Encode categorical features
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# --- Sidebar: Model Selection ---
st.sidebar.header("3. Model")
model_name = st.sidebar.selectbox(
    "Choose a model",
    ["K-Nearest Neighbors", "Decision Tree", "Logistic Regression", "Random Forest"]
)

# --- Sidebar: Hyperparameters ---
st.sidebar.header("4. Hyperparameters")
if model_name == "K-Nearest Neighbors":
    k = st.sidebar.slider("Number of neighbors (k)", 1, 25, 5)
    weights = st.sidebar.selectbox("Weight function", ["uniform", "distance"])
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
elif model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
elif model_name == "Logistic Regression":
    C = st.sidebar.select_slider(
        "Regularization strength (C)",
        options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        value=1.0
    )
    max_iter = st.sidebar.slider("Max iterations", 100, 2000, 500)
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
else:
    n_estimators = st.sidebar.slider("Number of trees", 10, 300, 100)
    max_depth_rf = st.sidebar.slider("Max depth", 1, 20, 5)
    min_samples_split_rf = st.sidebar.slider("Min samples split", 2, 20, 2)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth_rf,
        min_samples_split=min_samples_split_rf,
        random_state=42
    )

# --- Sidebar: Train/Test Split ---
st.sidebar.header("5. Train / Test Split")
test_size = st.sidebar.slider("Test set size (%)", 10, 50, 20) / 100
scale_features = st.sidebar.checkbox("Scale features (StandardScaler)", value=True)

# --- Train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
if scale_features:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Results ---
st.subheader(f"Model: {model_name} — Results")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric(
    "Precision (macro)",
    f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.3f}"
)
col3.metric(
    "Recall (macro)",
    f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.3f}"
)
col4.metric(
    "F1 Score (macro)",
    f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.3f}"
)

st.divider()

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax_cm, colorbar=False)
plt.tight_layout()
st.pyplot(fig_cm)

st.divider()

# --- ROC Curve (binary or multi-class OvR) ---
if hasattr(model, "predict_proba"):
    st.subheader("ROC Curve")
    y_prob = model.predict_proba(X_test)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"Class {le.classes_[i]} (AUC={roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_roc)

st.divider()

# --- Feature Importance (tree-based) ---
if model_name in ["Decision Tree", "Random Forest"]:
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(7, max(3, len(feature_cols) * 0.35)))
    sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax_fi, palette="viridis")
    ax_fi.set_title("Feature Importances")
    plt.tight_layout()
    st.pyplot(fig_fi)
    st.divider()

# --- Hyperparameter Tuning Explorer ---
if model_name == "K-Nearest Neighbors":
    st.subheader("Hyperparameter Tuning: k vs Accuracy")
    k_range = range(1, min(31, len(X_train)))
    train_scores, test_scores = [], []
    for ki in k_range:
        m = KNeighborsClassifier(n_neighbors=ki, weights=weights)
        m.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, m.predict(X_train)))
        test_scores.append(accuracy_score(y_test, m.predict(X_test)))
    fig_k, ax_k = plt.subplots(figsize=(7, 4))
    ax_k.plot(list(k_range), train_scores, label="Train Accuracy", marker="o", markersize=3)
    ax_k.plot(list(k_range), test_scores, label="Test Accuracy", marker="o", markersize=3)
    ax_k.axvline(x=k, color="red", linestyle="--", label=f"Selected k={k}")
    ax_k.set_xlabel("k")
    ax_k.set_ylabel("Accuracy")
    ax_k.set_title("KNN: k vs Accuracy")
    ax_k.legend()
    plt.tight_layout()
    st.pyplot(fig_k)

elif model_name == "Decision Tree":
    st.subheader("Hyperparameter Tuning: Max Depth vs Accuracy")
    depth_range = range(1, 21)
    train_d, test_d = [], []
    for d in depth_range:
        m = DecisionTreeClassifier(max_depth=d, random_state=42)
        m.fit(X_train, y_train)
        train_d.append(accuracy_score(y_train, m.predict(X_train)))
        test_d.append(accuracy_score(y_test, m.predict(X_test)))
    fig_d, ax_d = plt.subplots(figsize=(7, 4))
    ax_d.plot(list(depth_range), train_d, label="Train Accuracy", marker="o", markersize=3)
    ax_d.plot(list(depth_range), test_d, label="Test Accuracy", marker="o", markersize=3)
    ax_d.axvline(x=max_depth, color="red", linestyle="--", label=f"Selected depth={max_depth}")
    ax_d.set_xlabel("Max Depth")
    ax_d.set_ylabel("Accuracy")
    ax_d.set_title("Decision Tree: Max Depth vs Accuracy")
    ax_d.legend()
    plt.tight_layout()
    st.pyplot(fig_d)

st.divider()
st.caption("Built with Streamlit + scikit-learn | Portfolio Update 3 | Vedanth Nandivada")
