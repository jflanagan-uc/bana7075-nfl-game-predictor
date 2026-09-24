# get_logos.py — run once:  pip install nfl_data_py requests
import os
import requests
import nfl_data_py as nfl
from config import LOGOS_DIR

os.makedirs(LOGOS_DIR, exist_ok=True)
teams = nfl.import_team_desc().drop_duplicates("team_nick")  # old codes (OAK, SD, STL) share nicknames

for _, t in teams.iterrows():
    path = os.path.join(LOGOS_DIR, f"{t['team_nick']}.png")
    r = requests.get(t["team_logo_espn"], timeout=15)
    if r.ok:
        with open(path, "wb") as f:
            f.write(r.content)
        print("saved", path)