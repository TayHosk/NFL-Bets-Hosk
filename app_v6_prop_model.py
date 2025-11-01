
# app_v6_prop_model.py
# NFL Player Prop Model – Google Sheets v6
# - loads 11 of Taylor's Google Sheets
# - supports MULTIPLE props per player
# - has debug sidebar
# - uses Plotly for charts
# - estimates over/under + anytime TD

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import plotly.express as px

st.set_page_config(page_title="NFL Player Prop Model (Sheets v6)", layout="centered")

# =========================================================
# 1) GOOGLE SHEETS (Taylor's exact 11)
# =========================================================
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


def find_player_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        low = col.strip().lower()
        if "player" in low or "name" in low:
            return col
    # fallback to 2nd column
    if len(df.columns) >= 2:
        return df.columns[1]
    return df.columns[0]


@st.cache_data(show_spinner=False)
def load_all_sheets():
    # returns a dict of all dfs
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

    # strip Team columns for consistency
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


data = load_all_sheets()

p_rec = data["p_rec"]
p_rush = data["p_rush"]
p_pass = data["p_pass"]
d_rb = data["d_rb"]
d_qb = data["d_qb"]
d_wr = data["d_wr"]
d_te = data["d_te"]

rec_player_col = find_player_col(p_rec)
rush_player_col = find_player_col(p_rush)
pass_player_col = find_player_col(p_pass)

# =========================================================
# SIDEBAR (DEBUG)
# =========================================================
with st.sidebar:
    st.header("🔎 Debug")
    st.write("Receiving player col:", rec_player_col)
    st.write("Rushing player col:", rush_player_col)
    st.write("Passing player col:", pass_player_col)
    st.write("Rows loaded:")
    st.write({
        "p_rec": len(p_rec),
        "p_rush": len(p_rush),
        "p_pass": len(p_pass),
        "d_rb": len(d_rb),
        "d_qb": len(d_qb),
        "d_wr": len(d_wr),
        "d_te": len(d_te),
    })
    st.write("If player is not found, check Google Sheet header names.")

# =========================================================
# UI
# =========================================================
st.title("🏈 NFL Player Prop Model (Sheets v6, Plotly)")
st.write("Select a player, opponent, and one or more props. The app will estimate the probability of hitting the over/under using your Google Sheets.")

player_name = st.text_input("Player name (must match your Google Sheet):")
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

# =========================================================
# PLAYER LOOKUP
# =========================================================
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
    st.error("❌ Player not found in any of the player sheets. Check spelling or column headers.")
    st.stop()


# =========================================================
# HELPER: pick defense df
# =========================================================
def pick_defense_df(prop_type: str, player_pos: str):
    if prop_type == "passing_yards":
        return d_qb
    if prop_type in ["rushing_yards", "carries"]:
        # QB rush uses QB defensive sheet; RB rush uses RB defensive sheet
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


# =========================================================
# HELPER: detect stat column from player_df
# =========================================================
def detect_stat_col(player_df: pd.DataFrame, prop_type: str):
    cols = [c.strip() for c in player_df.columns]
    low = [c.lower() for c in cols]

    mapping = {
        "receiving_yards": [
            "receiving_yards_total",
            "receiving_yards_per_game",
            "rec_yards",
            "rec_yds",
        ],
        "receptions": [
            "receiving_receptions_total",
            "receptions",
            "rec",
        ],
        "targets": [
            "receiving_targets_total",
            "targets",
        ],
        "rushing_yards": [
            "rushing_yards_total",
            "rushing_yards_per_game",
            "rush_yards",
            "rush_yds",
        ],
        "carries": [
            "rushing_carries_total",
            "rushing_carries_per_game",
            "rush_att",
            "rush_attempts",
        ],
        "passing_yards": [
            "passing_yards_total",
            "passing_yards_per_game",
            "pass_yards",
            "pass_yds",
        ],
    }

    candidates = mapping.get(prop_type, [])
    for cand in candidates:
        if cand in low:
            return cols[low.index(cand)]
    return None


# =========================================================
# CORE MODEL
# =========================================================
def run_prop_model(player_df, player_pos, prop_type, opponent_team):
    defense_df = pick_defense_df(prop_type, player_pos)
    if defense_df is None:
        return None

    stat_col = detect_stat_col(player_df, prop_type)
    if stat_col is None:
        return None

    # merge on Opponent if exists
    if "Opponent" in player_df.columns:
        merged = player_df.merge(defense_df, left_on="Opponent", right_on="Team", how="left")
    else:
        # no opponent history → just use player stats
        merged = player_df.copy()
        merged["Team"] = opponent_team

    # rolling avg of last 3
    merged["rolling_avg_3"] = merged[stat_col].rolling(3, min_periods=1).mean()
    season_avg = merged[stat_col].mean()

    # pick a defense col
    def_cols = [c for c in defense_df.columns if "Allowed" in c or "allowed" in c]
    def_col = def_cols[0] if def_cols else None

    X_cols = ["rolling_avg_3"]
    if def_col:
        X_cols.append(def_col)

    X = merged[X_cols].fillna(0)
    y = merged[stat_col].fillna(0)

    if len(X) < 2:
        return None

    model = LinearRegression()
    model.fit(X, y)

    # get opponent row
    opp_row = defense_df[defense_df["Team"].str.lower() == opponent_team.lower()]
    if opp_row.empty:
        return None

    last3 = merged[stat_col].tail(3).mean()
    feat = [last3]
    if def_col:
        feat.append(float(opp_row.iloc[0][def_col]))
    pred_next = model.predict([feat])[0]

    # rmse
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds)) if len(y) > 2 else max(5.0, np.std(y))

    return {
        "stat_col": stat_col,
        "last3": last3,
        "season_avg": season_avg,
        "pred_next": float(pred_next),
        "rmse": float(rmse),
        "merged": merged,
    }


def estimate_anytime_td(player_df, player_pos, opponent_team):
    # pick defense
    if player_pos == "QB":
        defense_df = d_qb
    elif player_pos == "RB":
        defense_df = d_rb
    elif player_pos == "TE":
        defense_df = d_te
    else:
        defense_df = d_wr

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

    td_series = player_df[td_col].fillna(0)
    y = (td_series > 0).astype(int)

    if "Opponent" in player_df.columns:
        merged = player_df.merge(defense_df, left_on="Opponent", right_on="Team", how="left")
    else:
        merged = player_df.copy()

    merged["td_rolling_3"] = td_series.rolling(3, min_periods=1).mean()
    X = merged[["td_rolling_3"]].fillna(0)

    if y.sum() == 0 or len(X) < 2:
        # fallback to player's historical TD rate
        return float(y.mean())

    lg = LogisticRegression()
    lg.fit(X, y)

    feat = [[td_series.tail(3).mean()]]
    prob = lg.predict_proba(feat)[0][1]
    return float(prob)


# =========================================================
# DISPLAY RESULTS
# =========================================================
st.header("📊 Results")

for prop in selected_props:
    if prop == "anytime_td":
        continue

    res = run_prop_model(player_df, player_pos, prop, opponent_team)
    if res is None:
        st.warning(f"Could not model **{prop}**. Check if the player sheet has the right column for that stat.")
        continue

    line_val = lines.get(prop, None)
    if line_val is None:
        continue

    pred = res["pred_next"]
    rmse = res["rmse"]
    # probability
    z = (line_val - pred) / rmse if rmse > 0 else 0
    prob_over = 1 - norm.cdf(z)
    prob_under = norm.cdf(z)

    st.subheader(f"Prop: {prop}")
    st.write(f"**Player:** {player_name}")
    st.write(f"**Target value (sportsbook line):** {line_val}")
    st.write(f"**Average last 3 games:** {res['last3']:.2f}")
    st.write(f"**Season average:** {res['season_avg']:.2f}")
    st.write(f"**Predicted stat:** {pred:.2f}")
    st.write(f"**Probability of the over:** {prob_over*100:.1f}%")
    st.write(f"**Probability of the under:** {prob_under*100:.1f}%")

    # Plotly bar chart
    fig_bar = px.bar(
        x=["Predicted", "Line"],
        y=[pred, line_val],
        title=f"{player_name} – {prop.replace('_', ' ').title()}",
        labels={"x": "Metric", "y": "Value"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Trend chart
    m = res["merged"]
    if res["stat_col"] in m.columns:
        fig_line = px.scatter(
            m.reset_index(),
            x=m.reset_index().index + 1,
            y=res["stat_col"],
            trendline="ols",
            title=f"{player_name} – {res['stat_col']} by game",
            labels={"x": "Game #", "y": res["stat_col"]},
        )
        st.plotly_chart(fig_line, use_container_width=True)

# Anytime TD
if "anytime_td" in selected_props:
    td_prob = estimate_anytime_td(player_df, player_pos, opponent_team)
    st.subheader("🔥 Anytime TD")
    if td_prob is not None:
        st.write(f"**Anytime TD probability:** {td_prob*100:.1f}%")
    else:
        st.write("Not enough TD data to estimate touchdown probability.")
