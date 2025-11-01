# app_v7_4_prop_model_tdDefenseAdj.py
# NFL Player Prop Model – Season Totals → Single-Game Projection
# Includes defensive-adjusted Anytime TD probability
# Taylor Hoskinson build v7.4

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Player Prop Model (v7.4 TD Adj)", layout="centered")

# -----------------------------------------------------------
# 1) Google Sheets
# -----------------------------------------------------------
SHEETS = {
    "total_offense": "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv",
    "total_passing": "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv",
    "total_rushing": "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv",
    "total_scoring": "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv",
    "player_receiving": "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv",
    "player_rushing": "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv",
    "player_passing": "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv",
    "def_rb": "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv",
    "def_qb": "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv",
    "def_wr": "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv",
    "def_te": "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv",
}

# -----------------------------------------------------------
# 2) Data loading utilities
# -----------------------------------------------------------
def normalize_header(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

def load_and_clean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
    elif "teams" in df.columns:
        df["team"] = df["teams"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_all():
    return {name: load_and_clean(url) for name, url in SHEETS.items()}

data = load_all()

p_rec, p_rush, p_pass = data["player_receiving"], data["player_rushing"], data["player_passing"]
d_rb, d_qb, d_wr, d_te = data["def_rb"], data["def_qb"], data["def_wr"], data["def_te"]

# -----------------------------------------------------------
# 3) Sidebar Debug
# -----------------------------------------------------------
with st.sidebar:
    st.header("🔎 Debug")
    st.write("Receiving cols:", list(p_rec.columns))
    st.write("Rushing cols:", list(p_rush.columns))
    st.write("Passing cols:", list(p_pass.columns))
    st.write("Def RB cols:", list(d_rb.columns))
    st.write("Def WR cols:", list(d_wr.columns))
    st.write("Def TE cols:", list(d_te.columns))
    st.write("Note: v7.4 adds Anytime TD defensive adjustment")

# -----------------------------------------------------------
# 4) Helper functions
# -----------------------------------------------------------
def find_player_in(df, name):
    if "player" not in df.columns:
        return None
    mask = df["player"].astype(str).str.lower() == name.lower()
    return df[mask].copy() if mask.any() else None

def detect_def_tds(def_df, opp_team):
    """Return rushing + receiving TDs allowed per game"""
    if def_df is None or def_df.empty:
        return 0.0
    df = def_df.copy()
    if "games_played" in df.columns:
        gp = df["games_played"].replace(0, np.nan)
    else:
        gp = 1.0

    td_cols = [c for c in df.columns if "td" in c and "allowed" in c]
    if not td_cols:
        return 0.0

    df["tds_pg"] = df[td_cols].sum(axis=1) / gp
    opp_row = df[df["team"].astype(str).str.lower() == opp_team.lower()]
    opp_avg = df["tds_pg"].mean()
    if not opp_row.empty:
        return float(opp_row["tds_pg"].iloc[0]), float(opp_avg)
    return float(opp_avg), float(opp_avg)

# -----------------------------------------------------------
# 5) UI Inputs
# -----------------------------------------------------------
st.title("🏈 NFL Player Prop Model (v7.4 – TD Defense Adj)")
st.write("Single-game projection engine using season totals and defensive context.")

player_name = st.text_input("Player name:")
opponent_team = st.text_input("Opponent team (match defense sheet name):")
prop_choices = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "targets",
    "carries",
    "anytime_td",
]
selected_props = st.multiselect("Select props:", prop_choices, default=["receiving_yards"])

lines = {}
for prop in selected_props:
    if prop != "anytime_td":
        lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=prop)

if not player_name or not opponent_team or not selected_props:
    st.stop()

st.header("📊 Results")

# -----------------------------------------------------------
# 6) Core Modeling Loop
# -----------------------------------------------------------
for prop in selected_props:

    # ----- ANYTIME TD BLOCK -----
    if prop == "anytime_td":
        st.subheader("🔥 Anytime TD (Offense + Defense Adjusted)")

        # find player across all 3 sheets
        player_df = find_player_in(p_rec, player_name) or find_player_in(p_rush, player_name) or find_player_in(p_pass, player_name)
        if player_df is None or player_df.empty:
            st.warning(f"❗ Player '{player_name}' not found in receiving, rushing, or passing sheets.")
            continue

        # collect TD columns (rushing + receiving)
        td_cols = [c for c in player_df.columns if "td" in c and "allowed" not in c]
        if not td_cols:
            st.warning("⚠️ No touchdown data found for this player.")
            continue

        gp = float(player_df.iloc[0].get("games_played", 1)) or 1.0
        total_tds = player_df[td_cols].iloc[0].astype(float).sum()
        td_pg = total_tds / gp

        # defensive context
        opp_rush_pg, league_rush_pg = detect_def_tds(d_rb, opponent_team)
        opp_rec_pg, league_rec_pg = detect_def_tds(d_wr, opponent_team)
        opp_te_pg, league_te_pg = detect_def_tds(d_te, opponent_team)

        opp_td_pg = np.nanmean([opp_rush_pg, opp_rec_pg, opp_te_pg])
        league_td_pg = np.nanmean([league_rush_pg, league_rec_pg, league_te_pg])
        adj_factor = opp_td_pg / league_td_pg if league_td_pg > 0 else 1.0
        adj_td_pg = td_pg * adj_factor
        prob_anytime = min(adj_td_pg, 1.0)

        st.write(f"**TD Columns Used:** {', '.join(td_cols)}")
        st.write(f"**Total TDs (season):** {total_tds:.1f}")
        st.write(f"**Games Played:** {gp:.0f}")
        st.write(f"**Player TDs/Game:** {td_pg:.2f}")
        st.write(f"**Defense TDs/Game (League Avg):** {league_td_pg:.2f}")
        st.write(f"**Opponent TDs/Game (Adjusted):** {opp_td_pg:.2f}")
        st.write(f"**Adjusted TDs/Game (Player × Defense):** {adj_td_pg:.2f}")
        st.write(f"**Estimated Anytime TD Probability:** {prob_anytime*100:.1f}%")

        bar_df = pd.DataFrame({
            "Category": ["Player TD Rate", "Adj. w/ Defense"],
            "TDs/Game": [td_pg, adj_td_pg]
        })
        fig_td = px.bar(bar_df, x="Category", y="TDs/Game",
                        title=f"{player_name} – Anytime TD Adjustment vs {opponent_team}")
        st.plotly_chart(fig_td, use_container_width=True)
        continue

    # ----- All other props stay identical to your v7.3 -----
    def find_sheet_for_prop(prop):
        if prop in ["receiving_yards", "receptions", "targets"]:
            return p_rec, "wr"
        elif prop in ["rushing_yards", "carries"]:
            return p_rush, "rb"
        elif prop == "passing_yards":
            return p_pass, "qb"
        return None, "wr"

    player_df, fallback_pos = find_sheet_for_prop(prop)
    player_df = find_player_in(player_df, player_name)
    if player_df is None or player_df.empty:
        st.warning(f"❗ {prop}: player '{player_name}' not found.")
        continue

    player_pos = player_df.iloc[0].get("position", fallback_pos)

    # stat col detection
    cols = list(player_df.columns)
    norm_cols = [normalize_header(c) for c in cols]
    stat_map = {
        "rushing_yards": ["rushing_yards_total", "rushing_yards_per_game"],
        "receiving_yards": ["receiving_yards_total", "receiving_yards_per_game"],
        "passing_yards": ["passing_yards_total", "passing_yards_per_game"],
        "receptions": ["receiving_receptions_total"],
        "targets": ["receiving_targets_total"],
        "carries": ["rushing_attempts_total"]
    }
    stat_col = None
    for cand in stat_map.get(prop, []):
        if cand in norm_cols:
            stat_col = cols[norm_cols.index(cand)]
            break
    if not stat_col:
        st.warning(f"⚠️ {prop}: no stat col found.")
        continue

    # convert to per-game
    season_val = float(player_df.iloc[0][stat_col])
    gp = float(player_df.iloc[0].get("games_played", 1)) or 1.0
    player_pg = season_val / gp

    # defense adjustment
    def_df = d_rb if player_pos == "rb" else (d_wr if player_pos == "wr" else d_qb)
    def_col = None
    norm = [normalize_header(c) for c in def_df.columns]
    if "yards_allowed" in "_".join(norm):
        for i, n in enumerate(norm):
            if "yards_allowed" in n:
                def_col = def_df.columns[i]
                break
    opp_row = def_df[def_df["team"].astype(str).str.lower() == opponent_team.lower()]
    if not opp_row.empty:
        opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0].get("games_played", 1))
    else:
        opp_allowed_pg = def_df[def_col].mean() / def_df["games_played"].mean()

    league_allowed_pg = (def_df[def_col] / def_df["games_played"]).mean()
    adj_factor = opp_allowed_pg / league_allowed_pg if league_allowed_pg > 0 else 1.0
    predicted_pg = player_pg * adj_factor

    # prob over/under
    line_val = lines.get(prop, 0.0)
    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = 1 - norm.cdf(z)
    prob_under = norm.cdf(z)

    st.subheader(prop.replace("_", " ").title())
    st.write(f"**Player (season):** {season_val:.2f} over {gp:.0f} games → {player_pg:.2f}/game")
    st.write(f"**Defense Used:** {def_col}")
    st.write(f"**Opponent Allowed/Game:** {opp_allowed_pg:.2f}")
    st.write(f"**Adj. Prediction:** {predicted_pg:.2f}")
    st.write(f"**Line:** {line_val}")
    st.write(f"**Over Probability:** {prob_over*100:.1f}% | **Under:** {prob_under*100:.1f}%")

    bar = px.bar(x=["Predicted", "Line"], y=[predicted_pg, line_val],
                 title=f"{player_name} – {prop.replace('_', ' ').title()}")
    st.plotly_chart(bar, use_container_width=True)
