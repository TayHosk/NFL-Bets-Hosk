# app_v7_prop_model.py
# NFL Player Prop Model – season-totals version
# - works with Taylor's 11 Google Sheets
# - NO per-game logs required
# - multi-prop
# - Plotly charts
# - debug sidebar
# - defensive adjustment based on "Allowed" columns

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import plotly.express as px

st.set_page_config(page_title="NFL Player Prop Model (Sheets v7)", layout="centered")

# =============================================================
# 1. GOOGLE SHEETS (your 11, same order as before)
# =============================================================
SHEET_TOTAL_OFFENSE = "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv"
SHEET_TOTAL_PASS_OFF = "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv"
SHEET_TOTAL_RUSH_OFF = "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv"
SHEET_TOTAL_SCORE_OFF = "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv"

SHEET_PLAYER_REC = "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv"
SHEET_PLAYER_RUSH = "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv"
SHEET_PLAYER_PASS = "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv"

SHEET_DEF_RB = "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv"
SHEET_DEF_QB = "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv"
SHEET_DEF_WR = "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv"
SHEET_DEF_TE = "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv"


def load_sheet(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


@st.cache_data(show_spinner=False)
def load_all():
    total_off = load_sheet(SHEET_TOTAL_OFFENSE)
    total_pass_off = load_sheet(SHEET_TOTAL_PASS_OFF)
    total_rush_off = load_sheet(SHEET_TOTAL_RUSH_OFF)
    total_score_off = load_sheet(SHEET_TOTAL_SCORE_OFF)

    p_rec = load_sheet(SHEET_PLAYER_REC)
    p_rush = load_sheet(SHEET_PLAYER_RUSH)
    p_pass = load_sheet(SHEET_PLAYER_PASS)

    d_rb = load_sheet(SHEET_DEF_RB)
    d_qb = load_sheet(SHEET_DEF_QB)
    d_wr = load_sheet(SHEET_DEF_WR)
    d_te = load_sheet(SHEET_DEF_TE)

    # clean team names
    for df in [d_rb, d_qb, d_wr, d_te]:
        if "Team" in df.columns:
            df["Team"] = df["Team"].astype(str).str.strip()

    return {
        "total_off": total_off,
        "total_pass_off": total_pass_off,
        "total_rush_off": total_rush_off,
        "total_score_off": total_score_off,
        "p_rec": p_rec,
        "p_rush": p_rush,
        "p_pass": p_pass,
        "d_rb": d_rb,
        "d_qb": d_qb,
        "d_wr": d_wr,
        "d_te": d_te,
    }


data = load_all()
p_rec = data["p_rec"]
p_rush = data["p_rush"]
p_pass = data["p_pass"]
d_rb = data["d_rb"]
d_qb = data["d_qb"]
d_wr = data["d_wr"]
d_te = data["d_te"]

# =============================================================
# 2. HELPER FUNCTIONS
# =============================================================
def find_player_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "player" in col.lower():
            return col
    # fallback to 2nd col
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]


rec_player_col = find_player_col(p_rec)
rush_player_col = find_player_col(p_rush)
pass_player_col = find_player_col(p_pass)

def pick_def_df(prop_type: str, player_pos: str):
    if prop_type == "passing_yards":
        return d_qb
    if prop_type in ["rushing_yards", "carries"]:
        if player_pos == "QB":
            return d_qb
        return d_rb
    if prop_type in ["receiving_yards", "receptions", "targets"]:
        if player_pos == "TE":
            return d_te
        if player_pos == "RB":
            return d_rb
        return d_wr
    return None

def find_defense_column(def_df: pd.DataFrame, prop_type: str):
    # try to pick the right "Allowed" column per prop
    cols = list(def_df.columns)
    low = [c.lower() for c in cols]

    # order matters: we pick the first one we find
    if prop_type == "rushing_yards" or prop_type == "carries":
        for cand in ["rushing_yards_allowed_total", "rushing_yards_allowed", "rush_yards_allowed", "rushing_attemps_allowed", "rushing_attempts_allowed"]:
            if cand in low:
                return cols[low.index(cand)]
    if prop_type == "receiving_yards" or prop_type == "receptions" or prop_type == "targets":
        for cand in ["receiving_yards_allowed_total", "receiving_yards_allowed", "receiving_receptions_allowed", "receiving_targets_allowed"]:
            if cand in low:
                return cols[low.index(cand)]
    if prop_type == "passing_yards":
        for cand in ["passing_yards_allowed_total", "passing_yards_allowed", "passing_completions_allowed", "passing_attempts_allowed"]:
            if cand in low:
                return cols[low.index(cand)]
    # fallback: any numeric "allowed"
    for i, c in enumerate(low):
        if "allowed" in c:
            return cols[i]
    return None

def detect_player_stat_col(player_df: pd.DataFrame, prop_type: str):
    cols = [c.strip() for c in player_df.columns]
    low = [c.lower() for c in cols]

    if prop_type == "rushing_yards":
        candidates = [
            "rushing_yards_total",
            "rushing yards total",
            "total rushing yards",
            "rush_yards",
            "rush yards",
            "rushing_yards_per_game",
        ]
    elif prop_type == "receiving_yards":
        candidates = [
            "receiving_yards_total",
            "receiving yards total",
            "total receiving yards",
            "rec_yards",
            "rec yards",
            "receiving_yards_per_game",
        ]
    elif prop_type == "passing_yards":
        candidates = [
            "passing_yards_total",
            "passing yards total",
            "total passing yards",
            "pass_yards",
            "pass yards",
            "passing_yards_per_game",
        ]
    elif prop_type == "receptions":
        candidates = [
            "receiving_receptions_total",
            "receptions",
            "rec",
        ]
    elif prop_type == "targets":
        candidates = [
            "receiving_targets_total",
            "targets",
        ]
    elif prop_type == "carries":
        candidates = [
            "rushing_attempts_total",
            "rushing_carries_total",
            "rushing attempts total",
            "rushing_carries_per_game",
        ]
    else:
        candidates = []

    for cand in candidates:
        if cand in low:
            return cols[low.index(cand)]

    # if not found, return None
    return None

def estimate_from_season(player_value: float, def_df: pd.DataFrame, def_col: str):
    """
    Very simple adjustment: scale player season value by
    (defense_allowed_per_game / league_allowed_per_game)
    """
    if def_df is None or def_col is None or def_col not in def_df.columns:
        # no defense → just return player season number
        return player_value, 10.0  # 10 = default stdev
    df = def_df.copy()
    # defense is season totals too
    # get league avg per game
    if "Games_Played" in df.columns:
        df["allowed_per_game"] = df[def_col] / df["Games_Played"].replace(0, np.nan)
    else:
        df["allowed_per_game"] = df[def_col]
    league_avg = df["allowed_per_game"].mean(skipna=True)
    opp_allowed = df["allowed_per_game"].iloc[0]  # we'll override later
    # we actually need the opponent row
    return player_value, league_avg  # will be overridden at runtime

def estimate_anytime_td_from_sheets(player_df, player_pos, def_df):
    # try to find player's TD column
    td_candidates = [
        "Receiving_TDs_Scored",
        "Rushing_TDs_Scored",
        "Passing_TDs_Scored",
        "TD",
        "Touchdowns",
    ]
    td_col = None
    for c in td_candidates:
        if c in player_df.columns:
            td_col = c
            break
    if td_col is None:
        return None
    td_val = float(player_df.iloc[0][td_col])
    games = float(player_df.iloc[0].get("Games_Played", 1))
    base_rate = td_val / games if games > 0 else 0.0

    # adjust by defense if available
    if def_df is not None:
        # try to find receiving or rushing TDs allowed
        def_td_col = None
        for c in def_df.columns:
            if "td" in c.lower() and "allowed" in c.lower():
                def_td_col = c
                break
        if def_td_col:
            # avg TDs allowed per game
            if "Games_Played" in def_df.columns:
                def_df["tds_allowed_pg"] = def_df[def_td_col] / def_df["Games_Played"].replace(0, np.nan)
                def_rate = def_df["tds_allowed_pg"].mean(skipna=True)
            else:
                def_rate = def_df[def_td_col].mean(skipna=True)
            # blend
            prob = 0.5 * base_rate + 0.5 * (def_rate / 1.5 if def_rate else 0)
            return max(0.0, min(prob, 1.0))
    return base_rate

# =============================================================
# 3. SIDEBAR DEBUG
# =============================================================
with st.sidebar:
    st.header("🔎 Debug")
    st.write("Receiving sheet player col:", rec_player_col)
    st.write("Rushing sheet player col:", rush_player_col)
    st.write("Passing sheet player col:", pass_player_col)
    st.write("Rows:", {
        "p_rec": len(p_rec),
        "p_rush": len(p_rush),
        "p_pass": len(p_pass),
        "d_rb": len(d_rb),
        "d_qb": len(d_qb),
        "d_wr": len(d_wr),
        "d_te": len(d_te),
    })
    st.write("Note: this version uses SEASON TOTALS, not per-game logs.")

# =============================================================
# 4. UI
# =============================================================
st.title("🏈 NFL Player Prop Model (Season Totals v7)")
st.write("This version is built for your current Google Sheets (season totals, not per-game).")

player_name = st.text_input("Player name (exactly as in your player sheets):")
opponent_team = st.text_input("Opponent team (must match 'Team' in defense sheets):")

prop_options = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "targets",
    "carries",
    "anytime_td",
]
selected_props = st.multiselect("Select props to evaluate", prop_options, default=["rushing_yards"])

lines = {}
for prop in selected_props:
    if prop == "anytime_td":
        continue
    lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=f"line_{prop}")

if not player_name or not opponent_team or not selected_props:
    st.stop()

# =============================================================
# 5. FIND PLAYER (season total, 1 row)
# =============================================================
player_df = None
player_pos = None

if player_name.lower() in p_rec[rec_player_col].astype(str).str.lower().values:
    player_df = p_rec[p_rec[rec_player_col].astype(str).str.lower() == player_name.lower()].copy()
    player_pos = player_df.iloc[0].get("Position", "WR")
elif player_name.lower() in p_rush[rush_player_col].astype(str).str.lower().values:
    player_df = p_rush[p_rush[rush_player_col].astype(str).str.lower() == player_name.lower()].copy()
    player_pos = player_df.iloc[0].get("Position", "RB")
elif player_name.lower() in p_pass[pass_player_col].astype(str).str.lower().values:
    player_df = p_pass[p_pass[pass_player_col].astype(str).str.lower() == player_name.lower()].copy()
    player_pos = "QB"
else:
    st.error("❌ Player not found in any player sheet. Check spelling or headers.")
    st.stop()

# =============================================================
# 6. PROCESS EACH PROP
# =============================================================
st.header("📊 Results")

for prop in selected_props:
    if prop == "anytime_td":
        continue

    # 1) detect player's stat column
    stat_col = detect_player_stat_col(player_df, prop)
    if stat_col is None:
        st.warning(f"⚠️ For **{prop}**, I could not find a matching player column in your sheet. Please make sure one of the standard names is used.")
        continue

    # player season value
    try:
        player_val = float(player_df.iloc[0][stat_col])
    except Exception:
        player_val = 0.0

    # 2) pick defense df
    def_df = pick_def_df(prop, player_pos)
    # pick opponent row
    def_row = None
    if def_df is not None and "Team" in def_df.columns:
        def_row = def_df[def_df["Team"].str.lower() == opponent_team.lower()]
        if def_row.empty:
            def_row = None

    # 3) get defense column
    def_col = find_defense_column(def_df, prop) if def_df is not None else None

    # 4) build prediction: player season × defense adjustment
    # compute opponent allowed per game
    if def_df is not None and def_col is not None:
        df = def_df.copy()
        if "Games_Played" in df.columns:
            df["allowed_pg"] = df[def_col] / df["Games_Played"].replace(0, np.nan)
        else:
            df["allowed_pg"] = df[def_col]
        league_avg = df["allowed_pg"].mean(skipna=True)
        if def_row is not None:
            if "Games_Played" in def_row.columns:
                opp_allowed_pg = float(def_row.iloc[0][def_col]) / float(def_row.iloc[0]["Games_Played"])
            else:
                opp_allowed_pg = float(def_row.iloc[0][def_col])
        else:
            opp_allowed_pg = league_avg
        # adjustment factor
        if league_avg and league_avg > 0:
            adj_factor = opp_allowed_pg / league_avg
        else:
            adj_factor = 1.0
    else:
        # no defense info
        adj_factor = 1.0
        league_avg = None

    predicted_stat = player_val * adj_factor

    # 5) get line
    line_val = lines.get(prop, None)
    if line_val is None:
        continue

    # 6) probability using simple stdev
    # no per-game → use 15% of predicted as stdev
    stdev = max(5.0, predicted_stat * 0.15)
    z = (line_val - predicted_stat) / stdev
    prob_over = 1 - norm.cdf(z)
    prob_under = norm.cdf(z)

    st.subheader(f"Prop: {prop}")
    st.write(f"**Player:** {player_name}")
    st.write(f"**Player season value (from sheet):** {player_val:.2f}")
    if league_avg is not None:
        st.write(f"**Defense adjustment:** x{adj_factor:.2f} vs league avg")
    st.write(f"**Predicted stat:** {predicted_stat:.2f}")
    st.write(f"**Line:** {line_val}")
    st.write(f"**Probability of the over:** {prob_over*100:.1f}%")
    st.write(f"**Probability of the under:** {prob_under*100:.1f}%")

    # Plotly bar
    fig_bar = px.bar(
        x=["Predicted", "Line"],
        y=[predicted_stat, line_val],
        title=f"{player_name} – {prop.replace('_', ' ').title()}",
        labels={"x": "Metric", "y": "Value"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Trendline placeholder (since no per-game logs)
    trend_df = pd.DataFrame({
        "Game": [1, 2, 3],
        prop: [player_val * 0.85, player_val, predicted_stat]
    })
    fig_line = px.line(
        trend_df,
        x="Game",
        y=prop,
        markers=True,
        title=f"{player_name} – {prop.replace('_', ' ').title()} (simulated trend)",
    )
    st.plotly_chart(fig_line, use_container_width=True)

# =============================================================
# 7. ANYTIME TD
# =============================================================
if "anytime_td" in selected_props:
    # pick correct defense
    if player_pos == "QB":
        def_df_td = d_qb
    elif player_pos == "RB":
        def_df_td = d_rb
    elif player_pos == "TE":
        def_df_td = d_te
    else:
        def_df_td = d_wr

    td_prob = estimate_anytime_td_from_sheets(player_df, player_pos, def_df_td)
    st.subheader("🔥 Anytime TD")
    if td_prob is not None:
        st.write(f"**Anytime TD probability:** {td_prob*100:.1f}%")
    else:
        st.write("Not enough TD data to estimate TD.")
