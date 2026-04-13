# Nandivada — Data Science Portfolio

**Vedanth Nandivada** | Applied Mathematics, University of Notre Dame

[![GitHub](https://img.shields.io/badge/GitHub-vnandiva-blue?logo=github)](https://github.com/vnandiva) [![Email](https://img.shields.io/badge/Email-vnandiva%40nd.edu-red)](mailto:vnandiva@nd.edu)

---

## About Me

I'm a junior at Notre Dame studying Applied Mathematics. I've been interning in business analytics and healthcare consulting (UnitedHealthGroup and West Monroe), and I'm using this portfolio to build out my data science skills alongside that work. I'm most interested in the parts of data science that actually produce something useful — clean pipelines, clear visualizations, tools people can actually use.

---

## Technical Skills

| Category | Tools & Technologies |
|---|---|
| **Programming Languages** | Python, SQL, R |
| **Data Analysis & Visualization** | Pandas, NumPy, Matplotlib, Seaborn, Tableau, Power BI |
| **Machine Learning** | Scikit-learn, Random Forest, Statistical Modeling, TensorFlow |
| **Tools & Platforms** | Git/GitHub, Jupyter Notebooks, Google Colab, Excel |
| **Statistical Methods** | EDA, Regression Analysis, Classification, Predictive Modeling |

## Portfolio Projects

### Project 1: Streamlit Interactive Data Explorer

**Folder:** [`basic_streamlit_app/`](https://github.com/vnandiva/Nandivada-Data-Science-Portfolio/blob/main/basic_streamlit_app)

**Overview:** A web app built with Streamlit where users can upload any CSV and immediately start exploring it — filters, summary stats, charts — without touching any code.

**What it demonstrates:**
- Building user-facing data tools with Python and Streamlit
- Dynamic UI components (dropdowns, sliders, filters)
- Real-time data visualization from user-uploaded data
- Clean application structure and deployment-ready code

**Key skills:** Python, Streamlit, Pandas, Matplotlib, interactive UI design

### Project 2: Federal R&D Budget — Tidy Data Analysis

**Folder:** [`TidyData-Project/`](https://github.com/vnandiva/Nandivada-Data-Science-Portfolio/blob/main/TidyData-Project)

**Overview:** A Jupyter Notebook project where I took a messy, wide-format dataset on U.S. Federal R&D spending (1976–2017) and transformed it into a clean, tidy structure using `pd.melt()`, `str.split()`, and `str.replace()`. After cleaning, I ran aggregations and built two visualizations to look at trends across agencies over time.

**What it demonstrates:**
- Spotting and fixing untidy data structures
- Using `melt()`, `str.split()`, and `str.replace()` to reshape messy data
- Aggregating with `groupby()` and `pivot_table()`
- Two visualizations (line chart + bar chart) built on the cleaned data
- Markdown writeups explaining each step

**Key skills:** Python, Pandas, Matplotlib, Seaborn, data wrangling, tidy data principles, EDA

**How it fits with the rest of the portfolio:** Project 1 is about the output — making data usable for people who aren't writing code. Project 2 is about the input — actually getting the raw data into a shape where it's worth doing anything with. They're two different ends of the same pipeline.

### Project 3: Interactive Machine Learning Explorer

**Folder:** [`MLStreamlitApp/`](https://github.com/vnandiva/Nandivada-Data-Science-Portfolio/tree/main/MLStreamlitApp)

**Live App:** [Launch on Streamlit Community Cloud](https://vnandiva-nandivada-data-science-por-mlstreamlitappapp-PLACEHOLDER.streamlit.app)

**Overview:** A fully interactive Streamlit app for supervised machine learning. Users can upload their own CSV dataset or choose from three built-in datasets (Iris, Wine, Breast Cancer), select from four classification models, tune hyperparameters via sliders, and immediately see the impact on model performance — including accuracy metrics, confusion matrix, ROC curve, and feature importances.

**What it demonstrates:**
- End-to-end ML pipeline in Python: data loading, preprocessing, model training, and evaluation
- Four supervised models: K-Nearest Neighbors, Decision Tree, Logistic Regression, Random Forest
- Hyperparameter tuning with real-time feedback including sweep charts showing accuracy across the full k or depth range
- Robust performance evaluation: accuracy, precision, recall, F1, confusion matrix, ROC/AUC
- Deployment to Streamlit Community Cloud with a public shareable URL

**Key skills:** Python, Streamlit, scikit-learn, Pandas, Matplotlib, Seaborn, model deployment

**How it fits with the overall portfolio:** Projects 1 and 2 built the foundation, i.e interactive data tools and clean data pipelines. Project 3 is where those come together with machine learning. It applies the interactive Python development from Project 1 to an ML context, building something that makes model experimentation genuinely accessible. Together, the three projects show I can take data from raw input to clean format to trained, evaluated, and deployed models.

## Repository Structure

```
.
├── MLStreamlitApp/        # Interactive ML Explorer — Streamlit app with 4 classifiers (Portfolio Update 3)
├── TidyData-Project/      # Tidy data analysis of U.S. Federal R&D budgets (Portfolio Update 2)
├── basic_streamlit_app/   # Interactive Streamlit data explorer (Portfolio Update 1)
├── Week 2/                # Weekly exercises
├── Week 3/                # Weekly exercises
├── Week 4/                # Weekly exercises
└── README.md              # This file
```

## Getting Started

Each project folder has:
- A Jupyter notebook or Python script with the full analysis
- A `README.md` with setup instructions and project notes
- The dataset (or a link to it)
- Saved visualizations (`.png` files)

To run locally:

```bash
git clone https://github.com/vnandiva/Nandivada-Data-Science-Portfolio.git
cd Nandivada-Data-Science-Portfolio/
pip install -r requirements.txt  # or see the project README for specific dependencies
jupyter notebook
```

## Contact

- **GitHub:** [github.com/vnandiva](https://github.com/vnandiva)
- **Email:** [vnandiva@nd.edu](mailto:vnandiva@nd.edu)

Last updated: April 2026
