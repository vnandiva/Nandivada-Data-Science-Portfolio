# MLStreamlitApp — Interactive Machine Learning Explorer

**Vedanth Nandivada** | Portfolio Update 3 | Applied Mathematics, University of Notre Dame

## Project Overview

This Streamlit app lets users experiment with supervised machine learning interactively — no coding required. You can upload your own CSV dataset or choose from three built-in sample datasets (Iris, Wine, Breast Cancer), select a model, tune hyperparameters via sliders, and immediately see how those choices affect model performance.

The goal was to build something genuinely useful: a tool that makes ML more accessible and that shows the relationship between hyperparameter choices and model behavior in real time.

**Live App:** Link will be updated after deployment to Streamlit Community Cloud.

## App Features

### Supported Models

| Model | Key Hyperparameters |
|---|---|
| K-Nearest Neighbors | k (neighbors), weight function, distance metric |
| Decision Tree | max depth, min samples split, criterion |
| Logistic Regression | regularization strength (C), max iterations, solver |
| Random Forest | number of trees, max depth, min samples split |

### What the App Shows

- **Dataset preview** — head of the uploaded/selected dataset with shape and missing value info
- **Accuracy, Precision, Recall, F1 Score** — displayed as metric cards
- **Confusion Matrix** — visual heatmap of predictions vs actuals
- **Feature Importances** — ranked bar chart for tree-based models (Decision Tree, Random Forest)
- **Prediction vs Actual table** — sample of 20 rows showing correct/incorrect predictions

### User Controls (Sidebar)

1. **Data** — choose a sample dataset (Iris, Wine, Breast Cancer) or upload your own CSV
2. **Features & Target** — select which columns to use as features and which is the target
3. **Train/Test Split** — set the test size percentage and random seed
4. **Model** — pick from 4 supervised classifiers
5. **Hyperparameters** — adjust model-specific parameters via sliders, radio buttons, and selectboxes

## Running Locally

```bash
# 1. Clone the portfolio repo
git clone https://github.com/vnandiva/Nandivada-Data-Science-Portfolio.git
cd Nandivada-Data-Science-Portfolio/MLStreamlitAPP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### Required Libraries

| Library | Version |
|---|---|
| streamlit | >=1.32.0 |
| pandas | >=2.0.0 |
| numpy | >=1.24.0 |
| scikit-learn | >=1.4.0 |
| matplotlib | >=3.7.0 |
| seaborn | >=0.13.0 |

## How Hyperparameters Are Selected / Tuned

Hyperparameters are exposed via Streamlit sidebar widgets:

- **Sliders** for continuous/integer params (k, max depth, number of trees, C, max iterations)
- **Radio buttons** for discrete choices (weight function, split criterion)
- **Selectbox** for solver and distance metric options

Changing any hyperparameter and re-training immediately updates all output metrics and visualizations, making it easy to see the impact of each setting.

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)
- [matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [seaborn Documentation](https://seaborn.pydata.org/)
