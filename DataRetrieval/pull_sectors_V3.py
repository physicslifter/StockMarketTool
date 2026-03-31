import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
from pdb import set_trace as st
import numpy as np
import re
import difflib

# Cached ticker->CIK map, loaded once on first call to get_sector.
_TICKER_CIK_MAP = None

# Cached ticker->company_name map, loaded from symbols.feather.
_SYMBOL_NAME_MAP = None

get_all = 1
test_ACC = 0
test_ADS = 0
test_AEB = 0
test_AHC = 0
test_BAF = 0
test_unknowns_0323 = 0
test_unknowns_0325 = 0

SYMBOLS_PATH = "../Data/symbols.feather"
MY_USER_AGENT = "Pat Gavin patpatpaddy@gmail.com"
df = pd.read_feather("../Data/all_ohlcv_no_ETFs.feather")


# ── helpers ───────────────────────────────────────────────────────────────────

def _score_match(cik: str, target_ticker: str, target_name: str, user_agent: str) -> float:
    """Returns a confidence score from 0.0 to 1.0. 1.0 guarantees the ticker belongs to this CIK."""
    time.sleep(0.12)
    try:
        url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # 1. Ticker Check (Highest Priority -> Score 1.0)
        edgar_tickers =[str(t).upper() for t in data.get("tickers", [])]
        tgt = str(target_ticker).upper()
        ticker_variants = {tgt, tgt.replace(".", "-"), tgt.replace("$", "-"), tgt.replace(".", ""), tgt.replace("$", "")}
        
        if any(v in edgar_tickers for v in ticker_variants):
            return 1.0  
            
        # 2. Name Similarity Scoring (Fallback)
        target_name_clean = _clean_company_name(target_name)
        edgar_name = data.get("name", "").upper()
        edgar_former_names =[fn.get("name", "").upper() for fn in data.get("formerNames", [])]
        
        names_to_check = [edgar_name] + edgar_former_names
        best_sim = 0.0
        
        for name_to_check in names_to_check:
            name_to_check_clean = _clean_company_name(name_to_check)
            if not target_name_clean or not name_to_check_clean: continue
            
            similarity = difflib.SequenceMatcher(None, target_name_clean, name_to_check_clean).ratio()
            is_substring = (target_name_clean in name_to_check_clean) or (name_to_check_clean in target_name_clean)
            
            # Substrings represent high confidence for heavily abbreviated dataset names
            if is_substring and similarity < 0.85:
                similarity = 0.85 
                
            if similarity > best_sim:
                best_sim = similarity
                
        return best_sim
            
    except Exception as e:
        print(f"      [X] Scoring failed due to network/parsing error: {e}")
        return 0.0

def _normalize_for_edgar(name: str) -> list[str]:
    variants =[]
    stripped = name.rstrip(". ,")
    variants.append(stripped)
    no_dots = stripped.replace(".", "")
    if no_dots != stripped:
        variants.append(no_dots)
    compressed = re.sub(r"\s+", " ", no_dots).strip()
    if compressed != no_dots:
        variants.append(compressed)
    return list(dict.fromkeys(variants))

def _clean_company_name(name: str) -> str:
    """Removes punctuation and generic corporate suffixes for highly accurate matching."""
    if not name: return ""
    name = str(name).upper()
    name = re.sub(r'[^\w\s]', '', name) # Remove punctuation
    name = re.sub(r'\s+', ' ', name).strip()
    
    suffixes =[' INC', ' INCORPORATED', ' CORP', ' CORPORATION', ' LLC', ' L L C', 
                ' LTD', ' LIMITED', ' PLC', ' COMPANY', ' CO', ' HOLDINGS', ' HOLDING', ' LP', ' L P', ' GROUP', ' BANCORP']
    
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
                changed = True
    return name

def _load_ticker_cik_map(user_agent: str) -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df = pd.DataFrame.from_dict(response.json(), orient="index")
    df["ticker"] = df["ticker"].str.upper()
    df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
    return df.set_index("ticker")

def _load_symbol_name_map(path: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["act_symbol"] = df["act_symbol"].str.upper()
    return df.set_index("act_symbol")

def _get_sic_from_cik(cik: str, user_agent: str) -> str | None:
    time.sleep(0.12) # Slight bump to prevent SEC 429 errors
    cik_padded = str(cik).zfill(10)
    # ... (keep your existing request / json loading code) ...
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    desc = data.get("sicDescription")
    if desc and str(desc).strip().upper() not in["UNKNOWN", "NONE", "0000", "SIC 0000", "NOT ASSIGNED", ""]:
        return str(desc).strip()
    sic = data.get("sic")
    if sic:
        sic_str = str(sic).strip().zfill(4)
        if sic_str == "0000":
            return None
            
        if sic_str == "6726": return "Unit Inv Trusts, Face-Amt Cert Co, And Closed-End Management Inv Co"
        if sic_str.startswith("67"): return "Investment Offices / Funds"
        # ... (keep your existing sic_str.startswith checks) ...
        return f"SIC {sic_str}"
        
    # FIX: If EDGAR leaves SIC blank (typical for ETFs/Trusts), infer from entity type/name
    name = str(data.get("name", "")).upper()
    if any(k in name for k in[" TRUST", " FUND", " ETF", " PORTFOLIO", " SERIES", " INVESTMENT"]):
        return "Investment Offices / Funds"
        
    return None

def _is_valid_match(cik: str, target_ticker: str, target_name: str, user_agent: str) -> bool:
    time.sleep(0.11)
    try:
        url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # 1. Ticker Check (with variants for share classes like AKO.A -> AKO-A)
        edgar_tickers =[str(t).upper() for t in data.get("tickers", [])]
        tgt = str(target_ticker).upper()
        ticker_variants = {tgt, tgt.replace(".", "-"), tgt.replace("$", "-"), tgt.replace(".", ""), tgt.replace("$", "")}
        
        if any(v in edgar_tickers for v in ticker_variants):
            return True
            
        # 2. Delisted/Renamed Check: Look at Current Name AND Former Names
        target_name_clean = _clean_company_name(target_name)
        edgar_name = data.get("name", "").upper()
        edgar_former_names =[fn.get("name", "").upper() for fn in data.get("formerNames", [])]
        
        names_to_check = [edgar_name] + edgar_former_names
        
        for name_to_check in names_to_check:
            name_to_check_clean = _clean_company_name(name_to_check)
            if not target_name_clean or not name_to_check_clean: continue
            
            is_substring = (target_name_clean in name_to_check_clean) or (name_to_check_clean in target_name_clean)
            similarity = difflib.SequenceMatcher(None, target_name_clean, name_to_check_clean).ratio()
            
            if similarity >= 0.75 or is_substring:
                print(f"      [✓] Validated: '{target_name}' matches EDGAR's '{name_to_check}' (Score: {round(similarity, 2)})")
                return True
                
        print(f"      [X] False positive avoided: '{target_name}' != EDGAR names: {names_to_check}")
        return False
            
    except Exception as e:
        print(f"      [X] Validation failed due to network/parsing error: {e}")
        return False

def _resolve_sector_via_company_name(ticker: str, company_name: str, user_agent: str) -> str | None:
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    
    def _edgar_name_search(name: str) -> list[str]:
        time.sleep(0.11)
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?company={requests.utils.quote(name)}"
            "&CIK=&type=&dateb=&owner=include&count=40"
            "&search_text=&action=getcompany&output=atom"
        )
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        if "xml" in response.headers.get("Content-Type", ""):
            root = ET.fromstring(response.content)
            for elem in root.iter():
                if elem.tag.endswith("cik") and elem.text:
                    return[elem.text.strip().zfill(10)]
        else:
            matches = re.findall(r'CIK=(\d{1,10})', response.text, re.IGNORECASE)
            seen = set()
            ciks =[]
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    ciks.append(m.zfill(10))
            return ciks[:3]
        return[]

    words = company_name.split()
    min_words = 1 if len(words) <= 1 else 2
    
    for i in range(len(words), min_words - 1, -1):
        candidate = " ".join(words[:i])
        for variant in _normalize_for_edgar(candidate):
            ciks = _edgar_name_search(variant)
            for cik in ciks:
                if not score_match(cik, ticker, company_name, user_agent):
                    continue
                try:
                    sic = _get_sic_from_cik(cik, user_agent)
                    if sic:
                        return sic
                except Exception: pass

    print(f"  [!] name search exhausted for '{company_name}', trying EFTS")
    if not words: return None
        
    search_variants =[]
    if len(words) >= 3: search_variants.append(" ".join(words[:3]))
    if len(words) >= 2: search_variants.append(" ".join(words[:2]))
    search_variants.append(words[0])

    for query in search_variants:
        query_clean = query.replace(".", " ")
        try:
            time.sleep(0.11)
            efts_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{requests.utils.quote(query_clean)}%22"
            resp = requests.get(efts_url, headers=headers)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits",[])
            
            for hit in hits:
                source = hit.get("_source", {})
                ciks = source.get("ciks",[])
                
                if ciks:
                    cik = str(ciks[0]).zfill(10)
                    if not _is_valid_match(cik, ticker, company_name, user_agent): continue
                    try:
                        sic = _get_sic_from_cik(cik, user_agent)
                        if sic: return sic
                    except Exception: pass
        except Exception: pass

    return None


# ── main sector resolver ──────────────────────────────────────────────────────

def get_sector(ticker: str, user_agent: str) -> str:
    global _TICKER_CIK_MAP, _SYMBOL_NAME_MAP

    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    ticker_upper = ticker.upper()

    # --- Strategy 1: browse-edgar atom ---
    cik_from_atom = None
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker_upper}&action=getcompany&output=atom"
    try:
        time.sleep(0.11)
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        if "xml" in response.headers.get("Content-Type", ""):
            root = ET.fromstring(response.content)
            for elem in root.iter():
                if elem.tag.endswith("assigned-sic-desc") and elem.text:
                    sic_desc = elem.text.strip()
                    if sic_desc.upper() not in["UNKNOWN", "NONE", "0000", "SIC 0000", "NOT ASSIGNED", ""]:
                        return sic_desc
            for elem in root.iter():
                if elem.tag.endswith("cik") and elem.text:
                    cik_from_atom = elem.text.strip()
                    break
            if cik_from_atom:
                print(f"[~] {ticker}: valid SIC missing in atom, trying submissions JSON (CIK {cik_from_atom})")
        else:
            print(f"  [!] {ticker}: EDGAR atom returned HTML, ticker not active directly in EDGAR index")
    except requests.exceptions.RequestException as e:
        print(f"  [X] {ticker}: network error on browse-edgar request: {e}")
    except ET.ParseError as e: pass

    # --- Strategy 2: submissions JSON using CIK from atom ---
    if cik_from_atom:
        try:
            sic_desc = _get_sic_from_cik(cik_from_atom, user_agent)
            if sic_desc: return sic_desc
            print(f"  [!] {ticker}: CIK {cik_from_atom} found but no valid SIC in submissions JSON")
        except requests.exceptions.RequestException: pass

    # --- Strategy 3: company_tickers.json map (With Format Variations) ---
    if _TICKER_CIK_MAP is None:
        try:
            _TICKER_CIK_MAP = _load_ticker_cik_map(user_agent)
            print(f"[+] Loaded EDGAR ticker-to-CIK map ({len(_TICKER_CIK_MAP)} entries)")
        except Exception: _TICKER_CIK_MAP = pd.DataFrame()

    # Generate ticker formats (AKO.A -> AKO-A) and base tickers (AKO) for preferred stocks
    ticker_variants =[
        ticker_upper,
        ticker_upper.replace(".", "-").replace("$", "-"),
        ticker_upper.replace(".", "").replace("$", "")
    ]
    base_ticker = re.split(r'[.$/-]', ticker_upper)[0]
    if base_ticker != ticker_upper: 
        ticker_variants.append(base_ticker)

    for tv in dict.fromkeys(ticker_variants):
        if tv in _TICKER_CIK_MAP.index:
            cik = _TICKER_CIK_MAP.loc[tv, "cik_str"]
            print(f"  [~] {ticker}: found in CIK map as '{tv}' (CIK {cik}), trying submissions")
            try:
                sic_desc = _get_sic_from_cik(cik, user_agent)
                if sic_desc: 
                    return sic_desc
                else:
                    print(f"  [!] {ticker}: CIK {cik} found but no valid SIC, trying next variant")
            except requests.exceptions.RequestException: pass

    # --- Strategy 4: company name search via symbols.feather ---
    if _SYMBOL_NAME_MAP is None:
        try:
            _SYMBOL_NAME_MAP = _load_symbol_name_map(SYMBOLS_PATH)
            print(f"[+] Loaded symbol name map ({len(_SYMBOL_NAME_MAP)} entries)")
        except Exception: _SYMBOL_NAME_MAP = pd.DataFrame()

    if ticker_upper in _SYMBOL_NAME_MAP.index:
        company_name = _SYMBOL_NAME_MAP.loc[ticker_upper, "company_name"]
        print(f"  [~] {ticker}: found in symbol map as '{company_name}', searching EDGAR by name")
        try:
            sic_desc = _resolve_sector_via_company_name(ticker_upper, company_name, user_agent)
            if sic_desc: return sic_desc
        except requests.exceptions.RequestException: pass

    # --- Strategy 5: yfinance final fallback (With Formats & Base Tickers) ---
    print(f"  [!] {ticker}: attempting yfinance fallback")
    try:
        import yfinance as yf
        
        # Build candidate list: exact ticker, standard format, and base parent ticker
        yf_candidates = [ticker_upper]
        if re.search(r'[.$/]', ticker_upper):
            yf_candidates.append(re.sub(r'[.$/]', '-', ticker_upper))
        if base_ticker and base_ticker != ticker_upper:
            yf_candidates.append(base_ticker)
            
        yf_candidates = list(dict.fromkeys(yf_candidates))
        
        # Test each independently to prevent exceptions from crashing the loop
        for cand in yf_candidates:
            try:
                info = yf.Ticker(cand).info
                sector = info.get("sector") or info.get("industry")
                if sector:
                    print(f"[~] {ticker}: resolved via yfinance fallback on '{cand}' -> {sector}")
                    return sector
            except Exception as e:
                print(f"      [X] yf.Ticker('{cand}') failed: {e}")
                continue
                
        print(f"  [!] {ticker}: yfinance returned no sector/industry info for any variant")
    except Exception as e:
        print(f"  [X] {ticker}: yfinance module error: {e}")

    return "Unknown"


# ── entry points ──────────────────────────────────────────────────────────────

if get_all == True:
    MY_USER_AGENT = "Pat Gavin patpatpaddy@gmail.com"
    df = pd.read_feather("../Data/all_ohlcv_no_ETFs.feather")
    sectors = []
    tickers =[]
    for c, ticker in enumerate(df.act_symbol.unique()):
        sector = get_sector(ticker, MY_USER_AGENT)
        tickers.append(ticker)
        sectors.append(sector)
        print(f"{np.round(100*c/12770, 2)}% complete", ticker, sector)
        time.sleep(0.15)

    sector_df = pd.DataFrame({"act_symbol": tickers, "sector": sectors})
    sector_df.to_csv("../Data/sectors_0325.csv")

if test_ACC == True:
    print(get_sector("ACC", MY_USER_AGENT))

if test_ADS == True:
    print(get_sector("ADS", MY_USER_AGENT))

if test_AEB == True:
    print(get_sector("AEB", MY_USER_AGENT))

if test_AHC == True:
    print(get_sector("AHC", MY_USER_AGENT))

if test_BAF == True:
    print(get_sector("BAF", MY_USER_AGENT))

if test_unknowns_0323 == True:
    tickers =["AINV", "AKO.A", "AKO.B", "ORCL$D"]
    for ticker in tickers:
        print(f"\n======\n{ticker}")
        print(get_sector(ticker, MY_USER_AGENT))
        print("==========")

if test_unknowns_0325 == True:
    MY_USER_AGENT = "Pat Gavin patpatpaddy@gmail.com"
    unknowns_df = pd.read_csv("../Data/unknown_20260324.csv")
    tickers = unknowns_df["act_symbol"].tolist()
    
    print(f"--- Testing {len(tickers)} Unknowns ---")
    results =[]
    
    for c, ticker in enumerate(tickers):
        sector = get_sector(ticker, MY_USER_AGENT)
        results.append({"act_symbol": ticker, "sector": sector})
        print(f"{np.round(100*c/len(tickers), 2)}% complete | Ticker: {ticker} | Sector: {sector}")
        time.sleep(0.15)
        
    out_df = pd.DataFrame(results)
    resolved = out_df[out_df.sector != "Unknown"]
    
    print("\n--- Final Results ---")
    print(f"Resolved: {len(resolved)} / {len(tickers)}")
    print("\nResolved Tickers:")
    print(resolved.to_string())
    
    print("\nEntering debugger via st(). Inspect `out_df` and `resolved`.")
    st()

print("Done")