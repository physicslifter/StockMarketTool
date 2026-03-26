'''
Pulls the symbol table from the local Dolt SQL database and saves it to
../Data/symbols.feather for use by get_sectors.py.

Run once (or whenever you want to refresh the symbol data):
    python3 fetch_symbols.py
'''

import pandas as pd
from mysql import connector as cnc
import re

def clean_company_name(name: str) -> str:
    # Suffixes to strip from the end (corporate legal suffixes)
    suffixes = [
        "Class A Common Stock", "Class B Common Stock",
        "Class C Common Stock", "Class D Common Stock",
        "Common Stock", "Ordinary Shares",
        "American Depositary Shares", "American Depositary Share",
        "Depositary Shares",
        ", Inc.", " Inc.", ", Inc", " Inc",
        ", Corp.", " Corp.", ", Corp", " Corp",
        ", Ltd.", " Ltd.", ", Ltd", " Ltd",
        ", LLC", " LLC", ", L.P.", " L.P.",
        ", LP", " LP", ", PLC", " PLC",
        ", N.A.", " N.A.", ", Co.", " Co.",
    ]

    # Keywords that signal the START of a security description.
    # Everything from the first match onward is stripped.
    security_markers = [
        r"\bPerp\b", r"\bPerpetual\b",
        r"\bFloating\b", r"\bFixed\b",
        r"\bPreferred\b", r"\bSenior\b", r"\bJunior\b",
        r"\bWarrants?\b", r"\bNotes?\b", r"\bBonds?\b",
        r"\bDebentures?\b", r"\bSecs?\b",
        r"\bSeries\s+[A-Z]\b",
        r"\b\d+[\.\d]*%",           # interest rates like "6.25%"
        r"\(.*\)$",                 # trailing parenthetical e.g. "(Netherlands)"
    ]

    cleaned = name.strip()

    # Pass 1: truncate at first security marker
    for marker in security_markers:
        match = re.search(marker, cleaned, re.IGNORECASE)
        if match:
            cleaned = cleaned[:match.start()].strip().rstrip(",. ").strip()

    # Pass 2: strip legal suffixes from the end
    for suffix in suffixes:
        if cleaned.upper().endswith(suffix.upper()):
            cleaned = cleaned[:-len(suffix)].strip().rstrip(",").strip()

    return cleaned


conn = cnc.connect(host="127.0.0.1",
                   port=3306,
                   user="root",
                   password="",
                   database=None)

query = "SELECT act_symbol, security_name FROM stocks.symbol;"
df = pd.read_sql(query, conn)
print(f"  {len(df)} rows fetched")

df["company_name"] = df["security_name"].apply(clean_company_name)

df.to_feather("../Data/symbols.feather")
print("Saved to ../Data/symbols.feather")
print(df.head(10).to_string(index=False))