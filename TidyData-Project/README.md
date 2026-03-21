# Federal R&D Budget: Tidy Data Project

**Course:** SP26-MDSC-20009-01 | **Assignment:** Portfolio Update 2

## Project Overview

This project applies **tidy data principles** (Wickham, 2014) to clean, reshape, and analyze U.S. Federal R&D budget data spanning 1976–2017. The raw dataset is in wide format with messy column headers that encode two variables at once (year + GDP). Using Python and pandas, we transform it into a tidy format suitable for analysis and visualization.

A **tidy dataset** satisfies three rules:
1. Each **variable** forms its own column.
2. Each **observation** forms its own row.
3. Each **type of observational unit** forms its own table.

The key transformation tools used are:
- `pd.melt()` — reshape wide to long format
- `str.split()` — extract year and GDP from combined column names
- `str.replace()` — clean trailing characters in numeric strings

---

## Dataset Description

- **Source:** Federal R&D Budgets, adapted from [TidyTuesday (2019-02-12)](https://github.com/rfordatascience/tidytuesday/tree/main/data/2019/2019-02-12)
- **File:** `data/fed_rd_year&gdp.csv`
- **Raw format:** 14 rows (one per federal department) × 43 columns (one per year, with GDP embedded in header)
- **Tidy format:** 562 rows × 4 columns: `department`, `year`, `rd_budget`, `gdp`
- **Coverage:** 14 federal agencies including DOD, HHS, NIH, NASA, NSF, DOE, and others
- **Pre-processing:** Column headers like `1976_gdp1790000000000.0` were split into separate `year` and `gdp` columns; rows with no budget data (e.g., DHS before 2002) were dropped.

---

## How to Run the Notebook

### Prerequisites

Install the required libraries:

```bash
pip install pandas matplotlib seaborn
```

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/vnandiva/Nandivada-Data-Science-Portfolio.git
   cd Nandivada-Data-Science-Portfolio/TidyData-Project
   ```

2. Open the notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
   Or open in [Google Colab](https://colab.research.google.com/).

3. Run all cells in order (Runtime > Run all in Colab).

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pandas` | >= 1.3 | Data loading, reshaping, aggregation |
| `matplotlib` | >= 3.4 | Plotting and chart formatting |
| `seaborn` | >= 0.11 | Plot styling |
| `io` | (stdlib) | Reading CSV from string |

---

## What the Notebook Does

1. **Load & inspect** the raw wide-format CSV
2. **Tidy the data** using `melt()`, `str.split()`, and `str.replace()`
3. **Verify** the tidy structure against Wickham's three rules
4. **Aggregate** with `groupby()` and `pivot_table()` to answer key questions
5. **Visualize** with two charts:
   - Line chart: R&D spending by top departments over time (1976–2017)
   - Horizontal bar chart: Average annual R&D budget by department

---

## Visualizations

### Visualization 1: R&D Spending Over Time
Line chart showing how the top 6 departments' R&D budgets changed year-by-year from 1976 to 2017. Highlights the Cold War era DOD surge and post-2000 HHS/NIH growth.

### Visualization 2: Average Annual R&D Budget by Department

![Average Annual R&D Budget by Department](viz2_avg_budget_by_dept.png)

Horizontal bar chart showing average annual R&D budget across all available years. DOD consistently dominates at $64.7B/year on average.

---

## Key Findings

- **DOD** dominates federal R&D spending, averaging **$64.7B/year** over the full period
- **HHS** and **NIH** saw explosive growth through the 2000s as biomedical research investment surged
- **NASA** peaked in the late 1970s–80s and declined post-Cold War
- The **2000s** were the highest-spending decade overall at $1.86 trillion in total R&D investment

---

## References

- Wickham, H. (2014). Tidy Data. *Journal of Statistical Software*, 59(10). https://vita.had.co.nz/papers/tidy-data.pdf
- Pandas Cheat Sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
- Dataset adapted from TidyTuesday: https://github.com/rfordatascience/tidytuesday/tree/main/data/2019/2019-02-12
