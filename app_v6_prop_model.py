# app_v7_4_prop_model_fixed_td.py
# NFL Player Prop Model – Season Totals → Single-Game Projection
# Adds defensive-adjusted Anytime TD probability
# Based on v7.3 stable build

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Player Prop Model (v7.4 TD Fixed)", layout="centered")

# ---------------------------------------------------------------------
# 1) Google Sheets
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# 2) Utility
# ---------------------------------------------------------------------
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

p_rec = data["player_receiving"]
p_rush = data["player_rushing"]
p_pass = data["player_passing"]
d_rb = data["def_rb"]
d_qb = data["def_qb"]
d_wr = data["def_wr"]
d_te = data["def_te"]

# ---------------------------------------------------------------------
# 3) Sidebar Debug
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Debug")
    st.write("Receiving cols:", list(p_rec.columns))
    st.write("Rushing cols:", list(p_rush.columns))
    st.write("Passing cols:", list(p_pass.columns))
    st.write("Def RB cols:", list(d_rb.columns))
    st.write("Def WR cols:", list(d_wr.columns))
    st.write("Def TE cols:", list(d_te.columns))
    st.write("Note: v7.4 adds fully fixed Anytime TD logic")

# ---------------------------------------------------------------------
# 4) Helpers
# ---------------------------------------------------------------------
def find_player_in(df: pd.DataFrame, player_name: str):
    if "player" not in df.columns:
        return None
    mask = df["player"].astype(str).str.lower() == player_name.lower()
    return df[mask].copy() if mask.any() else None

# ---------------------------------------------------------------------
# 5) UI
# ---------------------------------------------------------------------
st.title("🏈 NFL Player Prop Model (v7.4 – Anytime TD Fixed)")
player_name = st.text_input("Player name:")
opponent_team = st.text_input("Opponent team (match defense sheets):")
prop_choices = [
    "passing_yards", "rushing_yards", "receiving_yards",
    "receptions", "targets", "carries", "anytime_td"
]
selected_props = st.multiselect("Select props:", prop_choices, default=["receiving_yards"])

lines = {}
for prop in selected_props:
    if prop != "anytime_td":
        lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=prop)

if not player_name or not opponent_team or not selected_props:
    st.stop()

st.header("📊 Results")

# ---------------------------------------------------------------------
# 6) Prop Logic
# ---------------------------------------------------------------------
for prop in selected_props:

    # --- ANYTIME TD FIXED ---
    if prop == "anytime_td":
        st.subheader("🔥 Anytime TD (Rushing + Receiving + Defense Adjusted)")

        rec_df = find_player_in(p_rec, player_name)
        rush_df = find_player_in(p_rush, player_name)
        pass_df = find_player_in(p_pass, player_name)

        # choose whichever is found first
        player_df = None
        for df in [rec_df, rush_df, pass_df]:
            if df is not None and not df.empty:
                player_df = df
                break

        if player_df is None:
            st.warning(f"❗ {player_name} not found in any sheet.")
            continue

        td_cols = [c for c in player_df.columns if "td" in c and "allowed" not in c]
        games_col = "games_played" if "games_played" in player_df.columns else None

        if not td_cols or not games_col:
            st.warning("⚠️ Not enough TD or games played data.")
            continue

        total_tds = 0.0
        for col in td_cols:
            try:
                total_tds += float(player_df.iloc[0][col])
            except Exception:
                pass

        gp = float(player_df.iloc[0][games_col]) or 1.0
        player_td_pg = total_tds / gp

        # Defensive context (RB/WR/TE)
        def_dfs = [d_rb.copy(), d_wr.copy(), d_te.copy()]
        for d in def_dfs:
            if "games_played" not in d.columns:
                d["games_played"] = 1
            d["tds_pg"] = (
                d[[c for c in d.columns if "td" in c and "allowed" in c]].sum(axis=1)
                / d["games_played"].replace(0, np.nan)
            )

        league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs])
        opp_td_pg = np.nanmean([
            d.loc[d["team"].astype(str).str.lower() == opponent_team.lower(), "tds_pg"].mean()
            for d in def_dfs
        ])
        if np.isnan(opp_td_pg):
            opp_td_pg = league_td_pg

        adj_factor = opp_td_pg / league_td_pg if league_td_pg > 0 else 1.0
        adj_td_rate = player_td_pg * adj_factor
        prob_anytime = min(adj_td_rate, 1.0)

        st.write(f"**TD Columns Used:** {', '.join(td_cols)}")
        st.write(f"**Total TDs (season):** {total_tds:.1f}")
        st.write(f"**Games Played:** {gp:.0f}")
        st.write(f"**Player TDs/Game:** {player_td_pg:.2f}")
        st.write(f"**Defense TDs/Game (League Avg):** {league_td_pg:.2f}")
        st.write(f"**Opponent TDs/Game (Adj):** {opp_td_pg:.2f}")
        st.write(f"**Adjusted Player TD Rate:** {adj_td_rate:.2f}")
        st.write(f"**Estimated Anytime TD Probability:** {prob_anytime*100:.1f}%")

        bar_df = pd.DataFrame({
            "Category": ["Player TD Rate", "Adj. vs Opponent"],
            "TDs/Game": [player_td_pg, adj_td_rate],
        })
        fig_td = px.bar(bar_df, x="Category", y="TDs/Game",
                        title=f"{player_name} – Anytime TD Adjustment vs {opponent_team}")
        st.plotly_chart(fig_td, use_container_width=True)
        continue

    # --- ALL OTHER PROPS (unchanged from v7.3) ---
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
        player_df = find_player_in(p_rec, player_name) or find_player_in(p_rush, player_name) or find_player_in(p_pass, player_name)
        fallback_pos = "wr"

    if player_df is None or player_df.empty:
        st.warning(f"❗ {prop}: player '{player_name}' not found in matching sheet.")
        continue

    player_pos = player_df.iloc[0].get("position", fallback_pos)
    stat_col = None
    cols = list(player_df.columns)
    norm = [normalize_header(c) for c in cols]
    if prop == "rushing_yards":
        pri = ["rushing_yards_total", "rushing_yards_per_game"]
    elif prop == "receiving_yards":
        pri = ["receiving_yards_total", "receiving_yards_per_game"]
    elif prop == "passing_yards":
        pri = ["passing_yards_total", "passing_yards_per_game"]
    elif prop == "receptions":
        pri = ["receiving_receptions_total"]
    elif prop == "targets":
        pri = ["receiving_targets_total"]
    elif prop == "carries":
        pri = ["rushing_attempts_total"]
    else:
        pri = []

    for cand in pri:
        if cand in norm:
            stat_col = cols[norm.index(cand)]
            break

    if not stat_col:
        st.warning(f"⚠️ For {prop}, no matching stat column found.")
        continue

    season_val = float(player_df.iloc[0][stat_col])
    games_played = float(player_df.iloc[0].get("games_played", 1)) or 1.0
    player_pg = season_val / games_played

    def_df = d_rb if player_pos == "rb" else (d_wr if player_pos == "wr" else d_qb)
    norm = [normalize_header(c) for c in def_df.columns]
    def_col = None
    for i, nc in enumerate(norm):
        if "yards_allowed" in nc:
            def_col = def_df.columns[i]
            break

    opp_row = def_df[def_df["team"].astype(str).str.lower() == opponent_team.lower()]
    if not opp_row.empty:
        opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0].get("games_played", 1))
    else:
        opp_allowed_pg = (def_df[def_col] / def_df["games_played"]).mean()

    league_allowed_pg = (def_df[def_col] / def_df["games_played"]).mean()
    adj_factor = opp_allowed_pg / league_allowed_pg if league_allowed_pg > 0 else 1.0
    predicted_pg = player_pg * adj_factor

    line_val = lines.get(prop, 0.0)
    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    prob_under = float(np.clip(norm.cdf(z), 0.001, 0.999))

    st.subheader(prop.replace("_", " ").title())
    st.write(f"**Player (season):** {season_val:.2f} over {games_played:.0f} games → {player_pg:.2f}/game")
    st.write(f"**Defense column used:** {def_col}")
    st.write(f"**Opponent allowed/game:** {opp_allowed_pg:.2f}")
    st.write(f"**Adjusted prediction:** {predicted_pg:.2f}")
    st.write(f"**Line:** {line_val}")
    st.write(f"**Probability of OVER:** {prob_over*100:.1f}% | **UNDER:** {prob_under*100:.1f}%")

    fig_bar = px.bar(x=["Predicted (this game)", "Line"], y=[predicted_pg, line_val],
                     title=f"{player_name} – {prop.replace('_', ' ').title()}")
    st.plotly_chart(fig_bar, use_container_width=True)
