# standard imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler  # need scaler for KNN/LR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris, load_wine, load_breast_cancer  # built-in sample datasets
import matplotlib.pyplot as plt
import seaborn as sns

# page setup
st.set_page_config(page_title="ML Explorer", page_icon="🤖", layout="wide")

st.title("🤖 Interactive Machine Learning Explorer")
st.write("Upload a CSV dataset (or use the built-in sample), select a supervised ML model, tune hyperparameters, and explore model performance — all without writing a single line of code!")
st.divider()

st.sidebar.header("📊 Dataset Configuration")

# cache these so they don't reload every time user changes something
@st.cache_data
def load_sample_iris():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

@st.cache_data
def load_sample_wine():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

@st.cache_data
def load_sample_cancer():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

# let user pick dataset source from sidebar
dataset_choice = st.sidebar.radio("Choose dataset source:", ["Sample Dataset (Iris)", "Sample Dataset (Wine)", "Sample Dataset (Breast Cancer)", "Upload Custom CSV"])

df = None

# load whichever dataset was selected
if dataset_choice == "Sample Dataset (Iris)":
    df = load_sample_iris()
    st.sidebar.success("✅ Loaded Iris dataset")
elif dataset_choice == "Sample Dataset (Wine)":
    df = load_sample_wine()
    st.sidebar.success("✅ Loaded Wine dataset")
elif dataset_choice == "Sample Dataset (Breast Cancer)":
    df = load_sample_cancer()
    st.sidebar.success("✅ Loaded Breast Cancer dataset")
else:
    # handle csv upload
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Custom dataset uploaded")

if df is not None:
    # quick preview of the data
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # show basic shape + missing val count
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    st.divider()

    # let user choose which col is target and which are features
    st.subheader("🎯 Feature & Target Selection")
    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select target column:", all_columns, index=len(all_columns)-1)
    feature_columns = [col for col in all_columns if col != target_column]
    selected_features = st.multiselect("Select feature columns:", feature_columns, default=feature_columns)

    if len(selected_features) == 0:
        st.warning("⚠️ Please select at least one feature column.")
        st.stop()

    st.divider()

    # train/test split controls
    st.subheader("✂️ Train-Test Split")
    test_size = st.slider("Test set size (%)", 10, 50, 20, step=5) / 100
    random_state = st.number_input("Random seed", value=42, step=1)

    X = df[selected_features]
    y = df[target_column]

    # encode target if it's a string (e.g. category labels)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    st.info(f"📊 Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    st.divider()

    # model + hyperparameter selection
    st.subheader("🔧 Model Selection & Hyperparameter Tuning")
    model_choice = st.selectbox("Choose a supervised learning model:", ["K-Nearest Neighbors (KNN)", "Decision Tree", "Logistic Regression", "Random Forest"])

    model = None

    if model_choice == "K-Nearest Neighbors (KNN)":
        st.write("**KNN Hyperparameters:**")
        n_neighbors = st.slider("Number of neighbors (k)", 1, 20, 5)
        weights = st.radio("Weight function:", ["uniform", "distance"])
        metric = st.selectbox("Distance metric:", ["euclidean", "manhattan", "minkowski"])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    elif model_choice == "Decision Tree":
        st.write("**Decision Tree Hyperparameters:**")
        max_depth = st.slider("Max depth", 1, 20, 5)
        min_samples_split = st.slider("Min samples to split", 2, 20, 2)
        criterion = st.radio("Split criterion:", ["gini", "entropy"])
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=int(random_state))

    elif model_choice == "Logistic Regression":
        st.write("**Logistic Regression Hyperparameters:**")
        C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, step=0.01)  # lower C = more regularization
        max_iter = st.slider("Max iterations", 100, 1000, 200, step=50)
        solver = st.selectbox("Solver:", ["lbfgs", "liblinear", "saga"])
        model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=int(random_state))

    elif model_choice == "Random Forest":
        st.write("**Random Forest Hyperparameters:**")
        n_estimators = st.slider("Number of trees", 10, 200, 100, step=10)
        max_depth = st.slider("Max depth", 1, 20, 10)
        min_samples_split = st.slider("Min samples to split", 2, 20, 2)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=int(random_state))

    st.divider()

    st.subheader("🚀 Train Model")
    if st.button("🔥 Train Model"):
        with st.spinner("Training in progress..."):
            # scale features — important for distance-based models like KNN and LR
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # fit only on train, transform both

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        st.success("✅ Model trained successfully!")
        st.divider()

        # eval metrics
        st.subheader("📈 Model Performance Metrics")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("Precision", f"{prec:.2%}")
        col3.metric("Recall", f"{rec:.2%}")
        col4.metric("F1 Score", f"{f1:.2%}")
        st.divider()

        # confusion matrix heatmap
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        st.divider()

        # feature importance only available for tree-based models
        if hasattr(model, 'feature_importances_'):
            st.subheader("📊 Feature Importance")
            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            ax_imp.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
            ax_imp.set_xlabel("Importance Score")
            ax_imp.set_title("Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig_imp)
            st.divider()

        # show sample of predictions vs actuals so user can see where it got things wrong
        st.subheader("📋 Prediction vs Actual (Sample)")
        comparison_df = pd.DataFrame({'Actual': y_test[:20], 'Predicted': y_pred[:20]})
        comparison_df['Match'] = comparison_df['Actual'] == comparison_df['Predicted']
        st.dataframe(comparison_df, use_container_width=True)

else:
    st.info("👈 Please select or upload a dataset from the sidebar to begin.")

st.divider()
st.caption("Built with Streamlit + scikit-learn | ML Explorer App | Vedanth Nandivada")
