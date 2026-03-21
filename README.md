# Nandivada — Data Science Portfolio

**Vedanth Nandivada** | Applied Mathematics, University of Notre Dame

[![GitHub](https://img.shields.io/badge/GitHub-vnandiva-blue?logo=github)](https://github.com/vnandiva) [![Email](https://img.shields.io/badge/Email-vnandiva%40nd.edu-red)](mailto:vnandiva@nd.edu)

---

## About Me

I am a junior at the University of Notre Dame pursuing Applied Mathematics with a passion for data science, analytics, and deriving actionable insights from complex datasets. With experience in business analytics and healthcare consulting through internships at UnitedHealthGroup and West Monroe, I am committed to developing robust data-driven solutions and communicating findings effectively to stakeholders.

---

## Technical Skills

| Category | Tools & Technologies |
|---|---|
| **Programming Languages** | Python, SQL, R |
| **Data Analysis & Visualization** | Pandas, NumPy, Matplotlib, Seaborn, Tableau, Power BI |
| **Machine Learning** | Scikit-learn, Random Forest, Statistical Modeling, TensorFlow |
| **Tools & Platforms** | Git/GitHub, Jupyter Notebooks, Google Colab, Excel |
| **Statistical Methods** | EDA, Regression Analysis, Classification, Predictive Modeling |

---

## Portfolio Projects

### Project 1: Streamlit Interactive Data Explorer

**Folder:** [`basic_streamlit_app/`](./basic_streamlit_app/)

**Overview:** An interactive web application built with Streamlit that allows users to upload any CSV dataset and explore it visually through dynamic filters, summary statistics, and customizable charts — all without writing a single line of code.

**What it demonstrates:**
- Building user-facing data tools with Python and Streamlit
- Dynamic UI components (dropdowns, sliders, filters)
- Real-time data visualization from user-uploaded data
- Clean application structure and deployment-ready code

**Key skills:** Python, Streamlit, Pandas, Matplotlib, interactive UI design

---

### Project 2: Federal R&D Budget — Tidy Data Analysis

**Folder:** [`TidyData-Project/`](./TidyData-Project/)

**Overview:** A Jupyter Notebook project that applies **tidy data principles** (Wickham, 2014) to clean, reshape, and analyze U.S. Federal R&D budget data spanning 1976–2017. The raw dataset is in messy wide format — column headers encode two variables at once (year + GDP). Using `pd.melt()`, `str.split()`, and `str.replace()`, the data is transformed into a fully tidy, analysis-ready format.

**What it demonstrates:**
- Identifying and correcting untidy data structures
- Applying `melt()`, `str.split()`, and `str.replace()` to reshape messy data
- Aggregating data with `groupby()` and `pivot_table()`
- Creating two polished visualizations (line chart + horizontal bar chart) from the cleaned data
- Thorough Markdown documentation explaining every step and design choice

**Key skills:** Python, Pandas, Matplotlib, Seaborn, data wrangling, tidy data principles, EDA

**How it complements the portfolio:**
While Project 1 (Streamlit) focuses on the *presentation layer* — building tools that let users interact with data visually — Project 2 focuses on the *foundation* that makes any good analysis possible: clean, well-structured data. Tidy data principles are not just academic; they are the prerequisite for every meaningful chart, model, or insight. Together, these two projects demonstrate both the front-end (interactive apps) and back-end (data cleaning and transformation) dimensions of data science work, showing a complete picture of the data pipeline from raw messy input to polished visual output.

---

## Repository Structure

```
.
├── TidyData-Project/        # Tidy data analysis of U.S. Federal R&D budgets (Portfolio Update 2)
├── basic_streamlit_app/     # Interactive Streamlit data explorer (Portfolio Update 1)
├── Week 2/                  # Weekly exercises
├── Week 3/                  # Weekly exercises
├── Week 4/                  # Weekly exercises
└── README.md                # This file
```

---

## Getting Started

Each project folder contains:
- A Jupyter notebook or Python script with full analysis and commented code
- A `README.md` explaining the project goals, setup instructions, and key findings
- Relevant datasets (or links to data sources)
- Saved visualizations (`.png` files)

To run any notebook locally:
```bash
git clone https://github.com/vnandiva/Nandivada-Data-Science-Portfolio.git
cd Nandivada-Data-Science-Portfolio/<project-folder>
pip install -r requirements.txt  # or see project README for dependencies
jupyter notebook
```

---

## Contact

- **GitHub:** [github.com/vnandiva](https://github.com/vnandiva)
- **Email:** [vnandiva@nd.edu](mailto:vnandiva@nd.edu)

*Last updated: March 2026*
