# NFL Player Prop Model (v7.2)

This app pulls 11 Google Sheets (season totals) and estimates player prop probabilities.

## Features
- Auto-cleans Google Sheets headers (so small spelling/spacing issues don’t break it)
- Works with season totals (no per-game logs needed)
- Multiple props per player (rushing, receiving, passing, receptions, targets, carries, anytime TD)
- Plotly charts
- Debug sidebar to see what columns were actually loaded

## Deploy
1. Create a GitHub repo
2. Add:
   - app_v7_2_prop_model.py
   - requirements.txt
   - README.md
3. Go to Streamlit Cloud → New App
4. Select your repo
5. Set **main file** to `app_v7_2_prop_model.py`
6. Deploy
