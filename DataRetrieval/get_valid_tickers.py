'''
Script which removes undesired tickers from the dataset
It saves the removed tickers to a new dataset
This is used in the SecurityFilter class in clean_analysis.py
This class now removes the following from the universe:
    ETP, 
    Equity WRT, 
    closed-end fund, 
    unit, 
    right, 
    MLP, 
    royalty trst, 
    Ltd part, 
    tracking stk, 
    NY reg shrs, 
    test tickers
'''
import pandas as pd
import re

meta = pd.read_feather("../Data/symbols.feather")
name = meta['security_name'].str.lower()

# === FLAGS ===
is_etf = (meta['is_etf'] == 1)
is_test = (meta['is_test_issue'] == 1)

# === TICKER-LEVEL CHECKS ===
has_dollar_sign = meta['act_symbol'].str.contains(r'\$', regex=True)
has_dot_p = meta['act_symbol'].str.endswith('.P')

# === MANUAL BLACKLIST ===
manual_blacklist = {'ASA', 'KMF', 'TVE'}
is_blacklisted = meta['act_symbol'].isin(manual_blacklist)

# === MANUAL WHITELIST ===
# Add PFBC to whitelist
manual_whitelist = {'APTS', 'PFBC'}
is_whitelisted = meta['act_symbol'].isin(manual_whitelist)

# === HARD REMOVE ===
# NOTE: \binvestment corp\b is NOT here — it is in SOFT remove below
hard_remove_patterns = [
    r'\bwarr+ants?\b',
    r'\bunits?\b',
    r'\brights?\b',
    r'\bpreferred\b',
    r'\bpfd\b',
    r'\b\d+[\.\d]*\s*%',
    r'\bperp\b',
    r'\bsubordinated\b',
    r'\bdebentures?\b',
    r'\bnotes?\b',
    r'when.issued',
    r'when.distributed',
    r'ex.distribution',
    r'\bsubordinate\b.*\bvoting\b',
    r'\bzones?\b',
    r'\b[ls]\.?p\.?\b',
    r'\bfund\b',
    r'beneficial interest',
    r'\belements\b',
    r'linked to',
    r'\bdue\s+\w+\s+\d{4}\b',
    r'\bbonuses\b',
    r'\bsubunits?\b',
    r'nextshares',
    r'paired shares',
    r'business development company',
    r'limited liability company',
    r'\btr\s+ctf\b',
    r'depository receipt',
    r'cmn shs of bi',
    r'\blimited partnership\b',
]
hard_remove_combined = '|'.join(hard_remove_patterns)
is_hard_remove = name.str.contains(hard_remove_combined, regex=True)

# === KEEP: explicit common stock / ADR / REIT ===
keep_patterns = [
    r'commo[nm]\s+st[co]+k',
    r'common shares?',
    r'ordinary shares?',
    r'american depositary',
    r'american depository',
    r'\badrs?\b',
    r'\bads\b',
    r'\breit\b',
]
keep_combined = '|'.join(keep_patterns)
is_explicit_keep = name.str.contains(keep_combined, regex=True)

# === SOFT REMOVE: removed UNLESS a keep keyword is present ===
soft_remove_patterns = [
    r'closed.end fund',
    r'\betfs?\b',
    r'\betns?\b',
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
    r'\bcap\b.*\bsecs\b',
    r'\bsail securities\b',
    r'\binvestment corp\b',         # here, NOT in hard remove
]
soft_remove_combined = '|'.join(soft_remove_patterns)
is_soft_remove = name.str.contains(soft_remove_combined, regex=True)

# === FINAL LOGIC ===
hard_block = (is_hard_remove | has_dollar_sign | has_dot_p | is_blacklisted) & ~is_whitelisted
remove = is_etf | is_test | hard_block | (is_soft_remove & ~is_explicit_keep)
keep = ~remove

valid_tickers = set(meta.loc[keep, 'act_symbol'])
removed_tickers = set(meta.loc[~keep, 'act_symbol'])

# ====================================================================
print("=" * 60)
print("CLASSIFICATION SUMMARY")
print("=" * 60)
print(f"Total: {len(meta)}")
print(f"Keeping: {keep.sum()}")
print(f"Removing: {(~keep).sum()}")

# ====================================================================
# TEST 1: Must keep
# ====================================================================
must_keep = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOG': 'Alphabet C',
    'GOOGL': 'Alphabet A', 'AMZN': 'Amazon', 'TSLA': 'Tesla',
    'V': 'Visa', 'T': 'AT&T', 'JPM': 'JPMorgan', 'BAC': 'Bank of America',
    'XOM': 'ExxonMobil', 'JNJ': 'J&J', 'MO': 'Altria', 'NEM': 'Newmont',
    'TSM': 'TSMC', 'BABA': 'Alibaba', 'AME': 'AMETEK', 'YUM': 'Yum!',
    'FL': 'Foot Locker', 'CCK': 'Crown Holdings', 'KMX': 'CarMax',
    'AGNC': 'AGNC Investment', 'APTS': 'Preferred Apartment',
}
print(f"\nTEST 1: Must-keep ({len(must_keep)})")
test1_pass = True
for ticker, company in must_keep.items():
    if ticker in valid_tickers:
        status = "PASS"
    elif ticker not in set(meta['act_symbol']):
        status = "SKIP"
    else:
        status = "FAIL"
        test1_pass = False
    print(f"  {ticker:8s} ({company:25s}) -> {status}")
print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

# ====================================================================
# TEST 2: Must remove
# ====================================================================
must_remove = {
    'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq 100 ETF', 'IWM': 'Russell 2000 ETF',
    'BAC$A': 'BofA Preferred A', 'BAC$B': 'BofA Preferred B',
    'ZVZZT': 'NASDAQ Test', 'ZJZZT': 'NASDAQ Test', 'NTEST': 'NYSE Test',
}
print(f"\nTEST 2: Must-remove ({len(must_remove)})")
test2_pass = True
for ticker, desc in must_remove.items():
    if ticker in removed_tickers:
        status = "PASS"
    elif ticker not in set(meta['act_symbol']):
        status = "SKIP"
    else:
        status = "FAIL"
        test2_pass = False
    print(f"  {ticker:8s} ({desc:25s}) -> {status}")
print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

# ====================================================================
# TEST 3: Coverage against price data
# ====================================================================
print(f"\nTEST 3: Coverage against price data")
try:
    prices = pd.read_feather("../Data/all_ohlcv.feather")
    price_tickers = set(prices['act_symbol'].unique())
    in_both = price_tickers & valid_tickers
    in_prices_not_symbols = price_tickers - set(meta['act_symbol'])
    in_prices_removed = price_tickers & removed_tickers
    print(f"  Price data tickers:           {len(price_tickers)}")
    print(f"  Overlap (kept & in prices):   {len(in_both)}")
    print(f"  In prices but removed:        {len(in_prices_removed)}")
    print(f"  In prices but not in symbols: {len(in_prices_not_symbols)}")
    print(f"  Coverage: {len(in_both) / len(price_tickers):.1%}")
    if len(in_prices_not_symbols) > 0:
        print(f"  WARNING: {len(in_prices_not_symbols)} price tickers not in symbol table")
        print(f"  Sample: {sorted(list(in_prices_not_symbols))[:15]}")
except FileNotFoundError:
    print("  SKIPPED (all_ohlcv.feather not found)")

# ====================================================================
# TEST 4: False positives
# ====================================================================
print(f"\nTEST 4: False positive check")

# Known intentional removals that say "common stock" — NOT false positives
# Add \bpfd\b to intentional removals in Test 4
intentional_removal_pattern = '|'.join([
    r'\bfund\b',
    r'\b[ls]\.?p\.?\b',
    r'\blimited partnership\b',
    r'\bpfd\b',                     # structured products like CORTS
])
# Build suspect list directly from removed data
removed_df = meta[remove].copy()
removed_name = removed_df['security_name'].str.lower()
removed_etf = (removed_df['is_etf'] == 1)
removed_test = (removed_df['is_test_issue'] == 1)

suspect_mask = (
    removed_name.str.contains(r'common stock', regex=True) &
    ~removed_name.str.contains(intentional_removal_pattern, regex=True) &
    ~removed_name.str.contains(r'\bwarr+ants?\b|\bunits?\b|\brights?\b|when.issued|when.distributed', regex=True) &
    ~removed_etf &
    ~removed_test
)
suspects = removed_df[suspect_mask]

if len(suspects) > 0:
    print(f"  WARNING: {len(suspects)} suspects:")
    for _, row in suspects.iterrows():
        n = row['security_name'].lower()
        hard_matches = [p for p in hard_remove_patterns if re.search(p, n)]
        soft_matches = [p for p in soft_remove_patterns if re.search(p, n)]
        print(f"    {row['act_symbol']:10s} hard={hard_matches} soft={soft_matches}")
        print(f"               {row['security_name'][:90]}")
else:
    print(f"  PASS: No common stocks incorrectly removed")

# ====================================================================
# TEST 5: Random sample
# ====================================================================
print(f"\nTEST 5: Random sample audit")
print(f"  15 KEPT:")
for _, row in meta[keep].sample(15, random_state=42).iterrows():
    print(f"    {row['act_symbol']:8s}  {row['security_name'][:80]}")
print(f"\n  15 REMOVED:")
for _, row in meta[remove].sample(15, random_state=42).iterrows():
    print(f"    {row['act_symbol']:8s}  {row['security_name'][:80]}")

# ====================================================================
print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"  Total tickers:    {len(meta)}")
print(f"  Keeping:          {keep.sum()}")
print(f"  Removing:         {(~keep).sum()}")
print(f"  Test 1 (keep):    {'PASS' if test1_pass else 'FAIL'}")
print(f"  Test 2 (remove):  {'PASS' if test2_pass else 'FAIL'}")
print(f"  Test 4 (false +): {'PASS' if len(suspects) == 0 else f'WARNING ({len(suspects)} suspects)'}")
print(f"\n  If all tests pass, integrate into SecurityTypeFilter.")

# At the bottom of name_sort_test.py, after all tests pass
valid_df = meta.loc[keep, ['act_symbol']].copy()
valid_df.to_feather("../Data/valid_equity_tickers.feather")
print(f"Saved {len(valid_df)} valid tickers to ../Data/valid_equity_tickers.feather")
