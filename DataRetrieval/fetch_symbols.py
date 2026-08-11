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
        r'\betn\b',                         # Exchange traded notes
        r'\betns\b',
        r'exchange traded',                  # "Exchange Traded Access Securities"
        r'\bipath\b',                        # iPath ETN products
        r'\bproshares\b',                    # Leveraged/inverse products
        r'\bultra\b',                        # UltraShort, UltraPro etc
        r'\bleverage\b|\bleveraged\b',
        r'\binverse\b',
        r'\bpfd\b',                          # Abbreviated preferred
        r'\bseries\s+[a-z]\b',              # "Series H", "Series B" (preferred)
        r'depositary sh',                    # "Depositary Shs Repstg..." (preferred DRs)
        r'\b\d+[\.\d]*\s*%',                # Interest rates like "6.75%" (bonds/preferred)
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


def classify_security(row):
    name = row['security_name'].lower()
    
    # --- REMOVE if flagged ---
    if row.get('is_etf') == 'Y':
        return False
    if row.get('is_test_issue') == 'Y':
        return False
    
    # --- REMOVE by name pattern ---
    remove_patterns = [
        r'\bwarrants?\b',
        r'\bunits?\b',
        r'\brights?\b',
        r'\bpreferred\b',
        r'\bpfd\b',
        r'\bnotes?\b',
        r'\bdebentures?\b',
        r'closed.end fund',
        r'\betfs?\b',
        r'\betns?\b',
        r'\bfund\b',
        r'\btrust\b',
        r'exchange traded',
        r'\bipath\b',
        r'\bproshares\b',
        r'\bdirexion\b',
        r'\bishares\b',
        r'\bpowershares\b',
        r'\bultra\b',
        r'\bleveraged?\b',
        r'\binverse\b',
        r'\bbear\b.*\bshares\b',
        r'\bbull\b.*\bshares\b',
        r'\b\d+x\b',
        r'depositary sh',
        r'\bdep\s+shs\b',
        r'\bseries\s+[a-z]\b',
        r'\b\d+[\.\d]*\s*%',
        r'when.issued',
        r'when.distributed',
        r'ex.distribution',
        r'\bsubordinate\b.*\bvoting\b',
    ]
    for pattern in remove_patterns:
        if re.search(pattern, name):
            return False
    
    # --- If it survived all removal checks, KEEP it ---
    return True


# ---------- Pull data ----------
conn = cnc.connect(host="127.0.0.1",
                   port=3306,
                   user="root",
                   password="",
                   database=None)

query = """
    SELECT act_symbol, security_name, is_etf, is_test_issue 
    FROM stocks.symbol;
"""
df = pd.read_sql(query, conn)
print(f"  {len(df)} rows fetched")

# ---------- Classify BEFORE cleaning ----------
df['is_valid_equity'] = df.apply(classify_security, axis=1)

print(f"\n--- Classification Summary ---")
print(f"  Keeping:  {df['is_valid_equity'].sum()}")
print(f"  Removing: {(~df['is_valid_equity']).sum()}")

# Spot check: uncategorized (not kept, not explicitly a known removal type)
name_lower = df['security_name'].str.lower()
known_remove = (
    name_lower.str.contains(r'\bwarrants?\b|\bunits?\b|\brights?\b|\bpreferred\b', regex=True) |
    name_lower.str.contains(r'\bnotes?\b|\bdebentures?\b|\betf\b|\bfund\b|\btrust\b', regex=True) |
    (df['is_etf'] == 'Y') |
    (df['is_test_issue'] == 'Y')
)
uncategorized = df[~df['is_valid_equity'] & ~known_remove]
print(f"  Uncategorized (inspect these): {len(uncategorized)}")
if len(uncategorized) > 0:
    print(uncategorized[['act_symbol', 'security_name']].sample(min(20, len(uncategorized))).to_string(index=False))

# ---------- Clean names ----------
df["company_name"] = df["security_name"].apply(clean_company_name)

# ---------- Save ----------
df.to_feather("../Data/symbols.feather")
print(f"\nSaved to ../Data/symbols.feather")
print(df.head(10).to_string(index=False))