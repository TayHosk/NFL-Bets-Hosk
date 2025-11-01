# app_v7_3_fixed_td.py
# NFL Player Prop Model – Season Totals → Single-Game Projection
# - per-prop sheet selection
# - converts season totals → per-game
# - adjusts by opponent defense per-game
# - compares to sportsbook single-game line
# - keeps Plotly + debug
# - FIXED ANYTIME TD LOGIC

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Player Prop Model (v7.3 Fixed TD)", layout="centered")

# 1) your sheets
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

def normalize_header(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

def load_and_clean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]
    # normalize team
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
    elif "teams" in df.columns:
        df["team"] = df["teams"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_all():
    return {name: load_and_clean(url) for name, url in SHEETS.items()}

data = load_all()

p_rec = data["player_receiving"]
p_rush = data["player_rushing"]
p_pass = data["player_passing"]
d_rb = data["def_rb"]
d_qb = data["def_qb"]
d_wr = data["def_wr"]
d_te = data["def_te"]

# sidebar debug
with st.sidebar:
    st.header("🔎 Debug")
    st.write("Receiving cols:", list(p_rec.columns))
    st.write("Rushing cols:", list(p_rush.columns))
    st.write("Passing cols:", list(p_pass.columns))
    st.write("Def RB cols:", list(d_rb.columns))
    st.write("Def WR cols:", list(d_wr.columns))
    st.write("Def TE cols:", list(d_te.columns))
    st.write("Note: v7.3 converts season totals → per-game before predicting.")

# helper: find player
def find_player_in(df: pd.DataFrame, player_name: str):
    if "player" not in df.columns:
        return None
    mask = df["player"].astype(str).str.lower() == player_name.lower()
    return df[mask].copy() if mask.any() else None

# detect stat column
def detect_stat_col(df: pd.DataFrame, prop: str):
    cols = list(df.columns)
    norm = [normalize_header(c) for c in cols]

    if prop == "rushing_yards":
        pri = ["rushing_yards_total", "rushing_yards_per_game", "rushingyards_total"]
        sec = ["rushing", "rush"]
    elif prop == "receiving_yards":
        pri = ["receiving_yards_total", "receiving_yards_per_game", "receivingyards_total"]
        sec = ["receiving", "rec"]
    elif prop == "passing_yards":
        pri = ["passing_yards_total", "passing_yards_per_game", "passingyards_total"]
        sec = ["passing"]
    elif prop == "receptions":
        pri = ["receiving_receptions_total", "receptions"]
        sec = ["receptions", "receivingreceptions"]
    elif prop == "targets":
        pri = ["receiving_targets_total", "targets"]
        sec = ["receivingtargets"]
    elif prop == "carries":
        pri = ["rushing_attempts_total", "rushing_carries_per_game"]
        sec = ["rushing_attempts", "rushingattempts", "rushingcarries"]
    else:
        return None

    # exact
    for cand in pri:
        if cand in norm:
            return cols[norm.index(cand)]
    # loose
    for cand in sec:
        for i, nc in enumerate(norm):
            if cand in nc:
                return cols[i]
    return None

# pick defense df
def pick_def_df(prop: str, pos: str):
    if prop == "passing_yards":
        return d_qb
    if prop in ["rushing_yards", "carries"]:
        return d_rb if pos != "qb" else d_qb
    if prop in ["receiving_yards", "receptions", "targets"]:
        if pos == "te":
            return d_te
        if pos == "rb":
            return d_rb
        return d_wr
    return None

# detect defense column
def detect_def_col(def_df: pd.DataFrame, prop: str):
    cols = list(def_df.columns)
    norm = [normalize_header(c) for c in cols]

    if prop in ["rushing_yards", "carries"]:
        prefs = ["rushing_yards_allowed_total", "rushing_yards_allowed", "rushing_attempts_allowed"]
    elif prop in ["receiving_yards", "receptions", "targets"]:
        prefs = ["receiving_yards_allowed_total", "receiving_yards_allowed", "receiving_receptions_allowed", "receiving_targets_allowed"]
    elif prop == "passing_yards":
        prefs = ["passing_yards_allowed_total", "passing_yards_allowed", "passing_completions_allowed", "passing_attempts_allowed"]
    else:
        prefs = []

    for cand in prefs:
        if cand in norm:
            return cols[norm.index(cand)]

    for i, nc in enumerate(norm):
        if "allowed" in nc:
            return cols[i]

    return None

# UI
st.title("🏈 NFL Player Prop Model (v7.3 Fixed TD)")
st.write("Projects single-game outcomes using season data + opponent defense.")

player_name = st.text_input("Player name:")
opponent_team = st.text_input("Opponent team (match defense sheets):")
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

for prop in selected_props:
    # ---- FIXED ANYTIME TD ----
    if prop == "anytime_td":
        st.subheader("🔥 Anytime TD")

        # Search across receiving → rushing → passing
        player_df = find_player_in(p_rec, player_name)
        if player_df is None or player_df.empty:
            player_df = find_player_in(p_rush, player_name)
        if player_df is None or player_df.empty:
            player_df = find_player_in(p_pass, player_name)

        if player_df is None or player_df.empty:
            st.warning(f"❗ {player_name} not found in any player sheet.")
            continue

        # Find all touchdown columns (exclude 'allowed')
        td_cols = [c for c in player_df.columns if "td" in c and "allowed" not in c]
        games_col = "games_played" if "games_played" in player_df.columns else None

        if td_cols and games_col:
            total_tds = player_df[td_cols].iloc[0].astype(float).sum()
            gp = float(player_df.iloc[0][games_col]) or 1.0
            td_pg = total_tds / gp
            prob_td = min(td_pg, 1.0)

            st.write(f"**TD columns used:** {', '.join(td_cols)}")
            st.write(f"**Total TDs (season):** {total_tds:.1f}")
            st.write(f"**Games Played:** {gp:.0f}")
            st.write(f"**TDs per game:** {td_pg:.2f}")
            st.write(f"**Estimated Anytime TD Probability:** {prob_td*100:.1f}%")
        else:
            st.warning("⚠️ No touchdown data found for this player.")
        continue

    # --- existing logic ---
    if prop in ["receiving_yards", "receptions", "targets"]:
        player_df = find_player_in(p_rec, player_name)
        fallback_pos = "wr"
    elif prop in ["rushing_yards", "carries"]:
        player_df = find_player_in(p_rush, player_name)
        fallback_pos = "rb"
    elif prop == "passing_yards":
        player_df = find_player_in(p_pass, player_name)
        fallback_pos = "qb"
    else:
        player_df = None
        fallback_pos = "wr"

    if player_df is None or player_df.empty:
        st.warning(f"❗ {prop}: player '{player_name}' not found in the matching sheet.")
        continue

    player_pos = player_df.iloc[0].get("position", fallback_pos)
    stat_col = detect_stat_col(player_df, prop)
    if not stat_col:
        st.warning(f"⚠️ For {prop}, no matching stat column found. Columns: {list(player_df.columns)}")
        continue

    season_val = float(player_df.iloc[0][stat_col])
    games_played = float(player_df.iloc[0].get("games_played", 1)) or 1.0
    player_pg = season_val / games_played

    def_df = pick_def_df(prop, player_pos)
    def_col = detect_def_col(def_df, prop) if def_df is not None else None

    opp_allowed_pg = None
    league_allowed_pg = None
    if def_df is not None and def_col is not None:
        if "games_played" in def_df.columns:
            league_allowed_pg = (def_df[def_col] / def_df["games_played"].replace(0, np.nan)).mean()
        else:
            league_allowed_pg = def_df[def_col].mean()

        opp_row = def_df[def_df["team"].astype(str).str.lower() == opponent_team.lower()]
        if not opp_row.empty:
            if "games_played" in opp_row.columns and float(opp_row.iloc[0]["games_played"]) > 0:
                opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played"])
            else:
                opp_allowed_pg = float(opp_row.iloc[0][def_col])
        else:
            opp_allowed_pg = league_allowed_pg

    if league_allowed_pg and league_allowed_pg > 0 and opp_allowed_pg is not None:
        adj_factor = opp_allowed_pg / league_allowed_pg
    else:
        adj_factor = 1.0

    predicted_pg = player_pg * adj_factor
    line_val = lines.get(prop, 0.0)
    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = 1 - norm.cdf(z)
    prob_under = norm.cdf(z)
    prob_over = float(np.clip(prob_over, 0.001, 0.999))
    prob_under = float(np.clip(prob_under, 0.001, 0.999))

    st.subheader(prop.replace("_", " ").title())
    st.write(f"**Player (season):** {season_val:.2f} over {games_played:.0f} games → **{player_pg:.2f} per game**")
    st.write(f"**Defense column used:** {def_col}")
    st.write(f"**Opponent allowed per game:** {opp_allowed_pg:.2f}" if opp_allowed_pg is not None else "**Opponent allowed per game:** n/a")
    st.write(f"**Adjusted prediction (this game):** {predicted_pg:.2f}")
    st.write(f"**Line:** {line_val}")
    st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
    st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

    fig_bar = px.bar(x=["Predicted (this game)", "Line"], y=[predicted_pg, line_val],
                     title=f"{player_name} – {prop.replace('_', ' ').title()}")
    st.plotly_chart(fig_bar, use_container_width=True)

    trend_df = pd.DataFrame({"Game": [1, 2, 3],
                             "Value": [player_pg * 0.8, player_pg, predicted_pg]})
    fig_line = px.line(trend_df, x="Game", y="Value", markers=True,
                       title=f"{player_name} – {prop.replace('_', ' ').title()} (simulated trend)")
    st.plotly_chart(fig_line, use_container_width=True)
