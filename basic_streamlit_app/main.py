from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Spotify Track Explorer", page_icon="🎵", layout="wide")

# streamlit sometimes runs from a different working directory, so I build the path from this file
DATA_PATH = Path(__file__).parent / "data" / "dataset.csv"

df = pd.read_csv(DATA_PATH)


df["duration_min"] = df["duration_ms"] / 60000



# header
st.title("🎵 Spotify Track Explorer")
st.write(
    "This app explores Spotify tracks and lets you filter by **genre**, **popularity**, "
    "and audio features like **danceability**, **energy**, **tempo**, and **duration**."
)

# quick info
st.caption(f"Loaded **{len(df):,}** tracks.")
genres = sorted(df["track_genre"].dropna().unique())
st.caption(f"Found **{len(genres):,}** genres.")

st.divider()

# preview
st.subheader("Dataset Preview")
st.dataframe(df.head(50), use_container_width=True, height=260)

st.divider()

# filters
st.sidebar.header("Filters")

selected_genres = st.sidebar.multiselect(
    "Genre",
    options=genres,
    default=[genres[0]] if len(genres) > 0 else []
)

pop_min, pop_max = st.sidebar.slider("Popularity", 0, 100, (0, 100))
dance_min, dance_max = st.sidebar.slider("Danceability", 0.0, 1.0, (0.0, 1.0), 0.01)
energy_min, energy_max = st.sidebar.slider("Energy", 0.0, 1.0, (0.0, 1.0), 0.01)

tempo_min_val = float(df["tempo"].min())
tempo_max_val = float(df["tempo"].max())
tempo_min, tempo_max = st.sidebar.slider(
    "Tempo (BPM)",
    float(round(tempo_min_val, 1)),
    float(round(tempo_max_val, 1)),
    (float(round(tempo_min_val, 1)), float(round(tempo_max_val, 1)))
)

dur_min_val = float(df["duration_min"].min())
dur_max_val = float(df["duration_min"].max())
dur_min, dur_max = st.sidebar.slider(
    "Duration (minutes)",
    float(round(dur_min_val, 2)),
    float(round(dur_max_val, 2)),
    (float(round(dur_min_val, 2)), float(round(dur_max_val, 2))),
    0.1
)

# filtering
filtered = df.copy()

if selected_genres:
    filtered = filtered[filtered["track_genre"].isin(selected_genres)]
else:
    filtered = filtered.iloc[0:0]

filtered = filtered[
    (filtered["popularity"].between(pop_min, pop_max))
    & (filtered["danceability"].between(dance_min, dance_max))
    & (filtered["energy"].between(energy_min, energy_max))
    & (filtered["tempo"].between(tempo_min, tempo_max))
    & (filtered["duration_min"].between(dur_min, dur_max))
]

# results
st.subheader("Filtered Results")
st.metric("Tracks matching filters", f"{len(filtered):,}")

cols_to_show = [
    "track_name", "artists", "album_name", "track_genre",
    "popularity", "danceability", "energy"
]

if len(filtered) == 0:
    st.warning("No tracks match the filters.")
    st.stop()

st.dataframe(filtered[cols_to_show].head(2000), use_container_width=True, height=420)

st.divider()