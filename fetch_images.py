"""Download one photo per dish from Wikipedia (skips images that already exist)."""
import time
import urllib.parse
from pathlib import Path

import requests

from menu_data import DISHES

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)
H = {"User-Agent": "MenuMoodSearch/1.0 (educational project)"}


def get(url, **kw):
    for attempt in range(6):
        r = requests.get(url, headers=H, timeout=30, **kw)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("rate limited")


missing = []
for dish_id, title, *_ in DISHES:
    path = OUT / f"{dish_id}.jpg"
    if path.exists():
        continue
    try:
        meta = get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)).json()
        path.write_bytes(get(meta["thumbnail"]["source"]).content)
        print("ok  ", dish_id)
    except Exception as e:
        print("FAIL", dish_id, title, e)
        missing.append(dish_id)
    time.sleep(3)

print("missing:", missing)
