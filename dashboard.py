"""Streamlit dashboard for predictions_upcoming.csv.   Run:  streamlit run dashboard.py"""
import os

import pandas as pd
import streamlit as st

from config import MODELS_PATH, LOGOS_DIR

st.set_page_config(page_title="NFL Game Predictions", layout="wide")

csv_path = os.path.join(MODELS_PATH, "predictions_upcoming.csv")
if not os.path.exists(csv_path):
    st.error("No predictions yet — run `python predict.py` first.")
    st.stop()

df = pd.read_csv(csv_path, parse_dates=["kickoff_ts"])
weeks = sorted(df["week"].unique())
week = st.selectbox("Week", weeks, index=0) if len(weeks) > 1 else weeks[0]
df = df[df["week"] == week].sort_values("confidence", ascending=False)

st.title(f"NFL Predictions — {int(df['season'].iloc[0])} Week {week}")
st.caption("Win probabilities from the trained model; kickoff times are US/Eastern.")


def logo(team, width=40):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if os.path.exists(path):
        st.image(path, width=width)


for i in range(0, len(df), 4):
    cols = st.columns(4)
    for col, (_, row) in zip(cols, df.iloc[i:i + 4].iterrows()):
        with col:
            st.markdown(f"<div style='color:gray;font-size:13px'>"
                        f"{row['kickoff_ts']:%a %b %d, %I:%M %p} ET</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 0.5, 1])
            with c1:
                logo(row["away_team"])
                st.write(f"{row['away_team']} {'✅' if row['pick'] == row['away_team'] else ''}")
                st.write(f"{row['away_win_prob']:.1%}")
            with c2:
                st.write("@")
            with c3:
                logo(row["home_team"])
                st.write(f"{row['home_team']} {'✅' if row['pick'] == row['home_team'] else ''}")
                st.write(f"{row['home_win_prob']:.1%}")
            st.caption(f"Confidence: {row['confidence']:.0%}")
            st.progress(float(min(1.0, row["confidence"])))
