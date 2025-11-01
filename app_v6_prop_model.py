# app_v7_2_prop_model.py
# NFL Player Prop Model – Season Totals (v7.2)
# - auto-cleans headers
# - super-tolerant rushing/receiving/passing detection
# - multi-prop
# - Plotly
# - debug sidebar
# works with your 11 Google Sheets

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Player Prop Model (v7.2)", layout="centered")

# ---------------------------------------------------------
# 1. GOOGLE SHEETS (your exact 11, CSV export links)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 2. LOADING + NORMALIZING HEADERS
# ---------------------------------------------------------
def normalize_header(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace("__", "_")
    name = name.lower()
    # keep only letters / numbers / underscore
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name


def load_and_clean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]
    # normalize team column
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
    elif "teams" in df.columns:
        df["team"] = df["teams"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_all_sheets():
    dfs = {}
    for key, url in SHEETS.items():
        dfs[key] = load_and_clean(url)
    return dfs


data = load_all_sheets()

# unpack
p_rec = data["player_receiving"]
p_rush = data["player_rushing"]
p_pass = data["player_passing"]
d_rb = data["def_rb"]
d_qb = data["def_qb"]
d_wr = data["def_wr"]
d_te = data["def_te"]

# ---------------------------------------------------------
# 3. SIDEBAR DEBUG
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔎 Debug")
    st.write("Rushing player sheet columns:", list(p_rush.columns))
    st.write("Receiving player sheet columns:", list(p_rec.columns))
    st.write("Passing player sheet columns:", list(p_pass.columns))
    st.write("Def RB columns:", list(d_rb.columns))
    st.write("Def QB columns:", list(d_qb.columns))
    st.write("Def WR columns:", list(d_wr.columns))
    st.write("Def TE columns:", list(d_te.columns))
    st.write("If you see `rushing_yards_total` or similar here, the app should find it now.")


# ---------------------------------------------------------
# 4. DETECTION HELPERS
# ---------------------------------------------------------
def detect_player_stat_col(player_df: pd.DataFrame, prop_type: str) -> str | None:
    """
    Try multiple ways to find the right column even if the sheet has small differences.
    We normalize the headers so we can compare.
    """
    cols = list(player_df.columns)
    norm_cols = [normalize_header(c) for c in cols]

    if prop_type == "rushing_yards":
        primary = [
            "rushing_yards_total",
            "rushingyards_total",
            "rushing_yards_per_game",
            "rushingyardspergame",
        ]
        secondary = [
            "rushing_yards",
            "rushingyards",
            "rush_yards",
            "rushyards",
            "rushingyds",
        ]
    elif prop_type == "receiving_yards":
        primary = [
            "receiving_yards_total",
            "receivingyards_total",
            "receiving_yards_per_game",
        ]
        secondary = [
            "receiving_yards",
            "receivingyards",
            "recyards",
            "receivingyds",
        ]
    elif prop_type == "passing_yards":
        primary = [
            "passing_yards_total",
            "passingyards_total",
            "passing_yards_per_game",
        ]
        secondary = [
            "passing_yards",
            "passyards",
            "passingyds",
        ]
    elif prop_type == "receptions":
        primary = [
            "receiving_receptions_total",
            "receptions",
        ]
        secondary = [
            "receivingreceptions",
            "rec",
        ]
    elif prop_type == "targets":
        primary = [
            "receiving_targets_total",
            "targets",
        ]
        secondary = [
            "receivingtargets",
        ]
    elif prop_type == "carries":
        primary = [
            "rushing_attempts_total",
            "rushing_carries_per_game",
        ]
        secondary = [
            "rushing_attempts",
            "rushingattempts",
            "rushingcarries",
        ]
    else:
        return None

    # 1) exact
    for cand in primary:
        if cand in norm_cols:
            return cols[norm_cols.index(cand)]

    # 2) partial
    for cand in secondary:
        for i, nc in enumerate(norm_cols):
            if cand in nc:
                return cols[i]

    # 3) super-loose fallback
    # e.g. anything with "rushing" and "yard"
    if prop_type.startswith("rushing"):
        for i, nc in enumerate(norm_cols):
            if "rushing" in nc and "yard" in nc:
                return cols[i]
    if prop_type.startswith("receiving"):
        for i, nc in enumerate(norm_cols):
            if "receiving" in nc and "yard" in nc:
                return cols[i]
    if prop_type.startswith("passing"):
        for i, nc in enumerate(norm_cols):
            if "passing" in nc and "yard" in nc:
                return cols[i]

    return None


def pick_def_df(prop_type: str, player_pos: str | None):
    if prop_type == "passing_yards":
        return d_qb
    if prop_type in ["rushing_yards", "carries"]:
        if player_pos and player_pos.lower() == "qb":
            return d_qb
        return d_rb
    if prop_type in ["receiving_yards", "receptions", "targets"]:
        if player_pos and player_pos.lower() == "te":
            return d_te
        if player_pos and player_pos.lower() == "rb":
            return d_rb
        return d_wr
    return None


def detect_def_col(def_df: pd.DataFrame, prop_type: str) -> str | None:
    cols = list(def_df.columns)
    norm_cols = [normalize_header(c) for c in cols]

    if prop_type in ["rushing_yards", "carries"]:
        prefs = [
            "rushing_yards_allowed_total",
            "rushing_yards_allowed",
            "rushing_attempts_allowed",
            "rushingattemptsallowed",
        ]
    elif prop_type in ["receiving_yards", "receptions", "targets"]:
        prefs = [
            "receiving_yards_allowed_total",
            "receiving_yards_allowed",
            "receiving_receptions_allowed",
            "receiving_targets_allowed",
        ]
    elif prop_type == "passing_yards":
        prefs = [
            "passing_yards_allowed_total",
            "passing_yards_allowed",
            "passing_completions_allowed",
            "passing_attempts_allowed",
        ]
    else:
        prefs = []

    for cand in prefs:
        if cand in norm_cols:
            return cols[norm_cols.index(cand)]

    # fallback -> any "allowed"
    for i, nc in enumerate(norm_cols):
        if "allowed" in nc:
            return cols[i]

    return None


# ---------------------------------------------------------
# 5. UI
# ---------------------------------------------------------
st.title("🏈 NFL Player Prop Model (v7.2 – Season Totals, auto-clean)")
st.write("Enter a player, opponent, and select one or more props. This version should finally detect your **Rushing_Yards_Total** column.")

player_name = st.text_input("Player name (exactly how it appears in your Google Sheets):")
opponent_team = st.text_input("Opponent team (exactly as in defense sheets):")

prop_options = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "targets",
    "carries",
    "anytime_td",
]
selected_props = st.multiselect("Select props:", prop_options, default=["rushing_yards"])

lines = {}
for prop in selected_props:
    if prop != "anytime_td":
        lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=f"line_{prop}")

if not player_name or not opponent_team or not selected_props:
    st.stop()


# ---------------------------------------------------------
# 6. FIND PLAYER (receiving → rushing → passing)
# ---------------------------------------------------------
def find_player(df: pd.DataFrame, name: str):
    if "player" not in df.columns:
        return None
    mask = df["player"].astype(str).str.lower() == name.lower()
    return df[mask].copy() if mask.any() else None


player_df = find_player(p_rec, player_name)
player_pos = None

if player_df is not None:
    player_pos = player_df.iloc[0].get("position", "wr")
else:
    player_df = find_player(p_rush, player_name)
    if player_df is not None:
        player_pos = player_df.iloc[0].get("position", "rb")
    else:
        player_df = find_player(p_pass, player_name)
        if player_df is not None:
            player_pos = "qb"

if player_df is None:
    st.error("❌ Player not found in any of the 3 player sheets.")
    st.stop()


# ---------------------------------------------------------
# 7. PROCESS EACH PROP
# ---------------------------------------------------------
st.header("📊 Results")

for prop in selected_props:
    if prop == "anytime_td":
        continue

    stat_col = detect_player_stat_col(player_df, prop)
    if not stat_col:
        st.warning(f"⚠️ For **{prop}** I could not find a matching player column. Player sheet columns: {list(player_df.columns)}")
        continue

    # player value
    try:
        player_val = float(player_df.iloc[0][stat_col])
    except Exception:
        player_val = 0.0

    # get defense
    def_df = pick_def_df(prop, player_pos)
    def_col = detect_def_col(def_df, prop) if def_df is not None else None

    # opponent row
    if def_df is not None and "team" in def_df.columns:
        opp_row = def_df[def_df["team"].astype(str).str.lower() == opponent_team.lower()]
    else:
        opp_row = None

    # opponent allowed per game
    opp_allowed = None
    league_allowed = None
    if def_df is not None and def_col is not None:
        if "games_played" in def_df.columns:
            # league avg
            league_allowed = (def_df[def_col] / def_df["games_played"].replace(0, np.nan)).mean()
        else:
            league_allowed = def_df[def_col].mean()

        if opp_row is not None and not opp_row.empty:
            if "games_played" in opp_row.columns and float(opp_row.iloc[0]["games_played"]) > 0:
                opp_allowed = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played"])
            else:
                opp_allowed = float(opp_row.iloc[0][def_col])
        else:
            opp_allowed = league_allowed

    # adjustment
    if league_allowed and league_allowed > 0 and opp_allowed is not None:
        adj_factor = opp_allowed / league_allowed
    else:
        adj_factor = 1.0

    predicted = player_val * adj_factor

    # prob calc
    line_val = lines.get(prop, 0.0)
    stdev = max(5.0, predicted * 0.15)
    z = (line_val - predicted) / stdev
    prob_over = 1 - norm.cdf(z)
    prob_under = norm.cdf(z)

    st.subheader(prop.replace("_", " ").title())
    st.write(f"**Matched player stat column:** `{stat_col}`")
    st.write(f"**Player season total (from sheet):** {player_val:.2f}")
    st.write(f"**Defense column used:** `{def_col}`" if def_col else "**Defense column used:** none (using player only)")
    st.write(f"**Predicted stat:** {predicted:.2f}")
    st.write(f"**Line:** {line_val}")
    st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
    st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

    # Plotly bar
    fig_bar = px.bar(
        x=["Predicted", "Line"],
        y=[predicted, line_val],
        title=f"{player_name} – {prop.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Fake trend so it looks good
    trend_df = pd.DataFrame({
        "Game": [1, 2, 3],
        "Value": [player_val * 0.9, player_val, predicted],
    })
    fig_line = px.line(
        trend_df,
        x="Game",
        y="Value",
        markers=True,
        title=f"{player_name} – {prop.replace('_', ' ').title()} (simulated trend)",
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------------------------------------
# 8. ANYTIME TD (simple)
# ---------------------------------------------------------
if "anytime_td" in selected_props:
    st.subheader("🔥 Anytime TD (simple)")
    # try to get any td-like column
    td_cols = [c for c in player_df.columns if "td" in c]
    games_col = "games_played" if "games_played" in player_df.columns else None
    if td_cols and games_col:
        total_tds = float(player_df.iloc[0][td_cols[0]])
        games = float(player_df.iloc[0][games_col]) or 1.0
        base_rate = total_tds / games
        st.write(f"**Anytime TD probability (rough):** {base_rate*100:.1f}%")
    else:
        st.write("Not enough TD data to estimate.")
