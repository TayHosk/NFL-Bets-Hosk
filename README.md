# NFL Player Prop Model (Season Totals v7)

This Streamlit app uses 11 Google Sheets (offense, player season totals, and defense vs position) to estimate NFL player prop probabilities.

## How it works
- Uses **season totals** only (no per-game logs required)
- Adjusts player season stat by opponent defensive strength
- Supports multiple props per player
- Shows Plotly charts

## Deploy
1. Create a GitHub repo
2. Upload:
   - app_v7_prop_model.py
   - requirements.txt
   - README.md
3. Go to Streamlit Cloud → New app
4. Select your repo
5. Set main file to `app_v7_prop_model.py`
6. Deploy
