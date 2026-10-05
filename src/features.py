import numpy as np
import pandas as pd

FEATURES = ["rank_diff", "rank_points_diff", "age_diff", "height_diff", "form_diff", "surface_form_diff", "h2h_a_winrate"]
DEFAULT_HEIGHT_CM = 180

def get_form(state, player_id, n: int = 10) -> float:
  hist = state.get(player_id, [])
  if not hist: return 0.5
  recent = hist[-n:]
  return sum(recent) / len(recent)

#Career win rate on a surface. Default is 0.5
def get_surface_winrate(state_surface, player_id, surface) -> float:
  hist = state_surface.get((player_id, surface), [])
  if not hist: return 0.5
  return sum(hist) / len(hist)

#Gets the win rate between the 2 players. Default is 0.5
def get_h2h_winrate(h2h_state, p1, p2) -> float:
  key = tuple(sorted((p1, p2)))
  record = h2h_state.get(key, {"wins": {p1: 0, p2: 0}})
  total = record["wins"].get(p1, 0) + record["wins"].get(p2, 0)
  if total == 0: return 0.5
  return record["wins"].get(p1, 0) / total

#Finds the features for player a and b
def make_feature_row(form_state, surface_state, h2h_state, surface, a: dict, b: dict) -> dict:
  a_ht = a["ht"] if pd.notna(a["ht"]) else DEFAULT_HEIGHT_CM
  b_ht = b["ht"] if pd.notna(b["ht"]) else DEFAULT_HEIGHT_CM
  return {
    "rank_diff": b["rank"] - a["rank"],
    "rank_points_diff": a["pts"] - b["pts"],
    "age_diff": a["age"] - b["age"],
    "height_diff": a_ht - b_ht,
    "form_diff": get_form(form_state, a["id"]) - get_form(form_state, b["id"]),
    "surface_form_diff": (get_surface_winrate(surface_state, a["id"], surface) - get_surface_winrate(surface_state, b["id"], surface)),
    "h2h_a_winrate": get_h2h_winrate(h2h_state, a["id"], b["id"]),
  }

def _side(m, prefix: str) -> dict:
  return {
    "id": m[f"{prefix}_id"], "rank": m[f"{prefix}_rank"],
    "pts": m[f"{prefix}_rank_points"], "age": m[f"{prefix}_age"],
    "ht": m[f"{prefix}_ht"],
  }


def build_dataset(matches: pd.DataFrame, seed: int = 999, verbose: bool = True):
  rng = np.random.default_rng(seed)
  form_state = {}       # player_id -> list of 0/1 (loss/win), chronological
  surface_state = {}    # (player_id, surface) -> list of 0/1
  h2h_state = {}        # sorted (id, id) -> {"wins": {pid: count}}
  rows = []

  for _, m in matches.iterrows():
    surface = m["surface"]
    winner, loser = _side(m, "winner"), _side(m, "loser")
    w_id, l_id = winner["id"], loser["id"]

    #Randomizes the order of the players (in the original dataset player A always wins)
    if rng.random() < 0.5: a, b, label = winner, loser, 1
    else: a, b, label = loser, winner, 0

    row = make_feature_row(form_state, surface_state, h2h_state, surface, a, b)
    row["label"] = label
    rows.append(row)

    form_state.setdefault(w_id, []).append(1)
    form_state.setdefault(l_id, []).append(0)
    surface_state.setdefault((w_id, surface), []).append(1)
    surface_state.setdefault((l_id, surface), []).append(0)
    key = tuple(sorted((w_id, l_id)))
    rec = h2h_state.setdefault(key, {"wins": {}})
    rec["wins"][w_id] = rec["wins"].get(w_id, 0) + 1

  dataset = pd.DataFrame(rows).dropna()
  if verbose: print(f"Feature rows built: {len(dataset)}")

  return dataset, form_state, surface_state, h2h_state
