import io
import pandas as pd
import requests

DATA_BASE_URL = (
  "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_{year}.csv"
)

KEEP_COLS = [
  "tourney_date", "surface", "winner_id", "winner_name", "winner_rank", "winner_rank_points", "winner_age", "winner_ht", "winner_hand",
  "loser_id", "loser_name", "loser_rank", "loser_rank_points", "loser_age", "loser_ht", "loser_hand",
]

#Finds data for a certain year
def find_season(year: int):
  resp = requests.get(DATA_BASE_URL.format(year=year), timeout=30)
  resp.raise_for_status()
  return pd.read_csv(io.StringIO(resp.text))

#Finds all matches from start year to end year
def find_matches(start_year: int, end_year: int, verbose: bool = False):
  frames = []
  tot_matches = 0
  for year in range(start_year, end_year + 1):
    try:
      df = find_season(year)
      df["season"] = year
      frames.append(df)
      if verbose:
        tot_matches += len(df)
        print(f"{len(df)} matches in {year}")
    except Exception as err:
      if verbose: print(f"No data found for {year}")
      pass
  if verbose: print(f"Total matches in {start_year}-{end_year}: {tot_matches}")
  return pd.concat(frames, ignore_index=True)
#Keeps rows with usable rank and surface data.
def clean_matches(raw: pd.DataFrame, verbose: bool = True):
  raw = raw.copy()
  raw["tourney_date"] = pd.to_datetime(raw["tourney_date"], format="%Y%m%d")
  #Keeping results the same every time the code executes
  raw = raw.sort_values("tourney_date", kind="stable").reset_index(drop=True)

  matches = raw[KEEP_COLS].dropna(
      subset=["winner_rank", "loser_rank", "surface"]
  ).reset_index(drop=True)

  if verbose: print(f"Matches with complete rank and surface data: {len(matches)}")

  return raw, matches
