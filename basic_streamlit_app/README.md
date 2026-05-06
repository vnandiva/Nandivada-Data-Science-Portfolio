# Spotify Track Explorer — Streamlit Data App

**Vedanth Nandivada** | Portfolio Update 1 | Applied Mathematics, University of Notre Dame

---

## Project Overview

This is an interactive web app built with Streamlit that lets you explore a Spotify tracks dataset. You can filter songs by genre, popularity, and audio features like danceability, energy, tempo, and duration — all through a clean sidebar UI, no coding required. The main goal was to practice building a user-facing data tool and get comfortable with Streamlit's components and layout.

---

## What It Demonstrates

- Building interactive data apps with Python and Streamlit
- Dynamic filtering with sidebar sliders and multiselect widgets
- Real-time data visualization using Plotly Express
- Working with Pandas for data loading and filtering
- Clean app structure with `st.set_page_config`, `st.divider`, `st.metric`

---

## Key Skills

Python, Streamlit, Pandas, Plotly Express, interactive UI design

---

## Features

- Filter tracks by **genre**, **popularity**, **danceability**, **energy**, **tempo**, and **duration**
- Dataset preview showing the first 50 rows
- Filtered results table (up to 2,000 tracks)
- Track count metric showing how many songs match current filters

---

## Libraries & Versions

| Library | Version | Purpose |
|---------|---------|--------|
| streamlit | >=1.32.0 | Web app framework and UI components |
| pandas | >=2.0.0 | Data loading, filtering, and manipulation |
| plotly | >=5.18.0 | Interactive charts and visualizations |

---

## How to Run

```bash
# Step 1
git clone https://github.com/vnandiva/Nandivada-Data-Science-Portfolio.git
# Step 2
cd Nandivada-Data-Science-Portfolio/basic_streamlit_app
# Step 3
pip install streamlit pandas plotly
streamlit run main.py
```

---

## Screenshots

*App UI showing sidebar filters and interactive chart output:*

> **Sidebar UI** — Genre multiselect, popularity slider, danceability/energy/tempo/duration range sliders. Filtering updates the dataset preview in real-time.

> **Chart Output** — Plotly Express scatter/bar charts render based on filtered results. Track count metric displayed at top. Up to 2,000 filtered tracks shown in sortable table.

*Note: Run locally with `streamlit run main.py` to see the interactive UI.*

---

## References

- [Streamlit Documentation](https://docs.streamlit.io/) — Official Streamlit API reference
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/) — Interactive chart library
- [Pandas Documentation](https://pandas.pydata.org/docs/) — Data manipulation library
- Spotify Tracks Dataset — Sourced from Kaggle (public dataset)
- SP26-MDSC-20009-01 Course Materials — University of Notre Dame

---

## Contact

- **GitHub:** [github.com/vnandiva](https://github.com/vnandiva)
- **Email:** [vnandiva@nd.edu](mailto:vnandiva@nd.edu)

---

*Last updated: May 2026*
