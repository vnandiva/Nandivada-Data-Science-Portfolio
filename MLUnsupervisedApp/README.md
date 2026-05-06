# Unsupervised Learning Explorer

**Author:** Vedanth Nandivada  
**University:** University of Notre Dame — Applied Mathematics  
**Course:** SP26-MDSC-20009-01  
**GitHub:** [Nandivada-Data-Science-Portfolio](https://github.com/vnandiva/Nandivada-Data-Science-Portfolio)

---

## Project Overview

An interactive Streamlit application for exploring unsupervised machine learning algorithms on real and uploaded datasets. Built with a Notre Dame color theme, this app provides intuitive controls for hyperparameter tuning and rich visualizations to support meaningful performance feedback.

**Streamlit App URL:** [Launch App](https://vnandiva-unsupervised-explorer.streamlit.app) — https://vnandiva-unsupervised-explorer.streamlit.app

---

## Algorithms Implemented

### 1. K-Means Clustering
- Interactive slider for number of clusters (k)
- Elbow plot (inertia vs. k)
- Silhouette score plot across k values
- 2D PCA scatter plot colored by cluster assignment
- Displays final silhouette score

### 2. Hierarchical Clustering
- Choice of linkage method (ward, complete, average, single)
- Truncated dendrogram with configurable sample size
- Cophenetic correlation coefficient
- Cluster assignments via tree cut
- 2D PCA scatter plot colored by cluster
- Silhouette score

### 3. PCA (Principal Component Analysis)
- Configurable number of principal components
- Scree plot with cumulative variance curve
- 2D PCA Scatter Plot
- 3D PCA Scatter Plot (when 3+ components selected)
- Component Loadings Heatmap showing feature contributions
- Optional coloring by true label (target excluded from features by default)

### 4. PCA + K-Means Combined
- PCA dimensionality reduction followed by K-Means in the reduced space
- Scree plot for PCA components
- Elbow plot and silhouette scores for K-Means in PCA space
- 2D scatter (PC1 vs PC2) colored by cluster
- Component Loadings Heatmap

---

## Dataset Support

- **Sample Datasets:** Iris, Wine, Breast Cancer (from scikit-learn)
- **Upload CSV:** Any clean CSV file with numeric features
- The `target` column is automatically excluded from features to prevent target leakage; it is available only for optional label coloring

---

## Hyperparameter Tuning

| Algorithm | Tunable Parameters |
|---|---|
| K-Means | Number of clusters (k), random state |
| Hierarchical | Linkage method, number of clusters, dendrogram sample size |
| PCA | Number of components, 2D vs 3D scatter |
| PCA + K-Means | PCA components, K-Means clusters (k), random state |

All sliders are bounded safely based on dataset size to prevent crashes on small datasets.

---

## Running Locally

```bash
# Clone the portfolio repository
git clone https://github.com/vnandiva/Nandivada-Data-Science-Portfolio.git
cd Nandivada-Data-Science-Portfolio/MLUnsupervisedApp

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Required Libraries

| Library | Version | Purpose |
|---|---|---|
| streamlit | >=1.28.0 | Web app framework |
| pandas | >=1.5.0 | Data manipulation |
| numpy | >=1.23.0 | Numerical operations |
| matplotlib | >=3.6.0 | Static plots |
| seaborn | >=0.12.0 | Heatmaps |
| scikit-learn | >=1.2.0 | ML algorithms |
| scipy | >=1.9.0 | Hierarchical clustering |
| plotly | >=5.10.0 | Interactive scatter plots |

---

## References

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [scipy Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Plotly Express Docs](https://plotly.com/python/plotly-express/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

*Built as part of the Notre Dame Data Science curriculum (SP26-MDSC-20009-01). All algorithms implemented using scikit-learn, scipy, and Streamlit.*


---

## App Screenshots

> *Live app: [https://vnandiva-unsupervised-explorer.streamlit.app](https://vnandiva-unsupervised-explorer.streamlit.app)*

**K-Means Clustering — Elbow Plot & Silhouette Scores**

![K-Means Dashboard](https://vnandiva-unsupervised-explorer.streamlit.app/~/+/media/kmeans.png)

*The K-Means section includes an interactive elbow plot (inertia vs. k), silhouette score chart, and a 2D PCA scatter plot colored by cluster assignment. Users tune the number of clusters via a sidebar slider.*

**Hierarchical Clustering — Dendrogram**

*The hierarchical clustering section renders a truncated dendrogram, computes the cophenetic correlation coefficient, and displays a 2D cluster scatter. Users choose the linkage method (ward, complete, average, single).*

**PCA — Scree Plot & Loadings Heatmap**

*The PCA section displays a scree plot with cumulative variance, 2D/3D scatter plots, and a component loadings heatmap. Optional coloring by true label is available.*

**To see all visualizations live, visit: [https://vnandiva-unsupervised-explorer.streamlit.app](https://vnandiva-unsupervised-explorer.streamlit.app)**
