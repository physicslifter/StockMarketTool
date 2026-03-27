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
get_missing = 0

SYMBOLS_PATH = "../Data/symbols.feather"
MY_USER_AGENT = "Pat Gavin patpatpaddy@gmail.com"

# ── helpers ───────────────────────────────────────────────────────────────────

def _map_sic_to_macro(sic_code: str | None) -> str:
    """Maps a 4-digit SIC code to the 10 official macro divisions."""
    if not sic_code or sic_code == "Unknown":
        return "Unknown"
    
    try:
        # Extract only digits in case it says "SIC 6726"
        sic_num_str = "".join(filter(str.isdigit, str(sic_code)))
        if not sic_num_str:
            return "Unknown"
            
        sic = int(sic_num_str)
        if 0 <= sic <= 999: return 'Agriculture'
        elif 1000 <= sic <= 1499: return 'Mining'
        elif 1500 <= sic <= 1799: return 'Construction'
        elif 2000 <= sic <= 3999: return 'Manufacturing'
        elif 4000 <= sic <= 4999: return 'Transport/Utilities'
        elif 5000 <= sic <= 5199: return 'Wholesale Trade'
        elif 5200 <= sic <= 5999: return 'Retail Trade'
        elif 6000 <= sic <= 6799: return 'Finance/RealEstate'
        elif 7000 <= sic <= 8999: return 'Services'
        elif 9000 <= sic <= 9999: return 'Public Admin'
    except Exception:
        pass
    return "Unknown"

def _text_to_macro(sector_str: str) -> str:
    """Fallback text mapper for yfinance sectors."""
    s = str(sector_str).upper()
    if "AGRICULTURE" in s: return "Agriculture"
    if any(x in s for x in["MINING", "BASIC MATERIALS", "ENERGY", "OIL", "GAS"]): return "Mining"
    if "CONSTRUCTION" in s: return "Construction"
    if any(x in s for x in["MANUFACTURING", "INDUSTRIALS", "DEFENSE", "CONSUMER DEFENSIVE", "CONSUMER CYCLICAL"]): return "Manufacturing"
    if any(x in s for x in["UTILITIES", "TRANSPORTATION", "COMMUNICATION"]): return "Transport/Utilities"
    if "WHOLESALE" in s: return "Wholesale Trade"
    if "RETAIL" in s: return "Retail Trade"
    if any(x in s for x in ["FINANCIAL", "REAL ESTATE"]): return "Finance/RealEstate"
    if any(x in s for x in ["HEALTHCARE", "TECHNOLOGY", "SERVICES"]): return "Services"
    return "Unknown"

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

def _get_sic_from_cik(cik: str, user_agent: str) -> tuple[str | None, str | None]:
    """Returns (sic_desc, sic_code)"""
    time.sleep(0.12)
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    sic_code = data.get("sic")
    sic_desc = data.get("sicDescription")
    
    if sic_code:
        sic_code = str(sic_code).strip().zfill(4)
        if sic_code == "0000":
            sic_code = None
            
    if sic_desc:
        sic_desc = str(sic_desc).strip()
        if sic_desc.upper() in["UNKNOWN", "NONE", "0000", "SIC 0000", "NOT ASSIGNED", ""]:
            sic_desc = None
            
    # If EDGAR leaves SIC blank, infer from entity type/name
    name = str(data.get("name", "")).upper()
    if any(k in name for k in[" TRUST", " FUND", " ETF", " PORTFOLIO", " SERIES", " INVESTMENT", " ETN"]):
        sic_code = sic_code or "6726"
        sic_desc = sic_desc or "Unit Inv Trusts, Face-Amt Cert Co, And Closed-End Management Inv Co"
        
    return sic_desc, sic_code

def _is_valid_match(cik: str, target_ticker: str, target_name: str, user_agent: str) -> bool:
    time.sleep(0.11)
    try:
        url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # 1. Ticker Check 
        edgar_tickers =[str(t).upper() for t in data.get("tickers", [])]
        tgt = str(target_ticker).upper()
        ticker_variants = {tgt, tgt.replace(".", "-"), tgt.replace("$", "-"), tgt.replace(".", ""), tgt.replace("$", "")}
        
        if any(v in edgar_tickers for v in ticker_variants):
            return True
            
        # 2. Delisted/Renamed Check
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

def _resolve_sector_via_company_name(ticker: str, company_name: str, user_agent: str) -> tuple[str|None, str|None]:
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
                if not _score_match(cik, ticker, company_name, user_agent):  # FIX: Added leading underscore
                    continue
                try:
                    desc, code = _get_sic_from_cik(cik, user_agent)
                    if desc:
                        return desc, code
                except Exception: pass

    print(f"  [!] name search exhausted for '{company_name}', trying EFTS")
    if not words: return None, None
        
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
                        desc, code = _get_sic_from_cik(cik, user_agent)
                        if desc: return desc, code
                    except Exception: pass
        except Exception: pass

    return None, None


# ── main sector resolver ──────────────────────────────────────────────────────

def get_sector(ticker: str, user_agent: str) -> tuple[str, str, str]:
    """Returns (sic_description, sic_code, macro_sector)"""
    global _TICKER_CIK_MAP, _SYMBOL_NAME_MAP

    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    ticker_upper = ticker.upper()

    # 0. Instantly fail fake exchange test tickers
    if any(x in ticker_upper for x in ["TEST", "EXIT", "IEXT", "XIET"]):
        return "Unknown", "Unknown", "Unknown"

    # --- Strategy 1: browse-edgar atom ---
    cik_from_atom = None
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker_upper}&action=getcompany&output=atom"
    try:
        time.sleep(0.11)
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        if "xml" in response.headers.get("Content-Type", ""):
            root = ET.fromstring(response.content)
            sic_code = None
            sic_desc = None
            
            for elem in root.iter():
                if elem.tag.endswith("assigned-sic") and elem.text:
                    sic_code = elem.text.strip()
                if elem.tag.endswith("assigned-sic-desc") and elem.text:
                    sic_desc = elem.text.strip()
                if elem.tag.endswith("cik") and elem.text:
                    cik_from_atom = elem.text.strip()
            
            if sic_desc and sic_desc.upper() not in["UNKNOWN", "NONE", "0000", "SIC 0000", "NOT ASSIGNED", ""]:
                return sic_desc, (sic_code or "Unknown"), _map_sic_to_macro(sic_code)
                
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
            desc, code = _get_sic_from_cik(cik_from_atom, user_agent)
            if desc: return desc, (code or "Unknown"), _map_sic_to_macro(code)
            print(f"  [!] {ticker}: CIK {cik_from_atom} found but no valid SIC in submissions JSON")
        except requests.exceptions.RequestException: pass

    # --- Strategy 3: company_tickers.json map (With Format Variations) ---
    if _TICKER_CIK_MAP is None:
        try:
            _TICKER_CIK_MAP = _load_ticker_cik_map(user_agent)
            print(f"[+] Loaded EDGAR ticker-to-CIK map ({len(_TICKER_CIK_MAP)} entries)")
        except Exception: _TICKER_CIK_MAP = pd.DataFrame()

    ticker_variants = [ticker_upper]
    ticker_variants.append(ticker_upper.replace(".", "-").replace("$", "-"))
    ticker_variants.append(ticker_upper.replace(".", "").replace("$", ""))
    
    base_ticker = re.split(r'[.$/-]', ticker_upper)[0]
    if base_ticker != ticker_upper: 
        ticker_variants.append(base_ticker)
        
    # NEW: Catch concatenated preferreds/rights (e.g., OXLCM -> OXLC, OFSSI -> OFS, BANXR -> BANX)
    if len(ticker_upper) == 5:
        ticker_variants.append(ticker_upper[:4]) # Try 4-letter base
        ticker_variants.append(ticker_upper[:3]) # Try 3-letter base

    for tv in dict.fromkeys(ticker_variants):
        if tv in _TICKER_CIK_MAP.index:
            cik = _TICKER_CIK_MAP.loc[tv, "cik_str"]
            print(f"  [~] {ticker}: found in CIK map as '{tv}' (CIK {cik}), trying submissions")
            try:
                desc, code = _get_sic_from_cik(cik, user_agent)
                if desc: return desc, (code or "Unknown"), _map_sic_to_macro(code)
                else: print(f"  [!] {ticker}: CIK {cik} found but no valid SIC, trying next variant")
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
            desc, code = _resolve_sector_via_company_name(ticker_upper, company_name, user_agent)
            if desc: return desc, (code or "Unknown"), _map_sic_to_macro(code)
        except requests.exceptions.RequestException: pass

    # --- Strategy 5: yfinance final fallback ---
    print(f"  [!] {ticker}: attempting yfinance fallback")
    try:
        import yfinance as yf
        yf_candidates = list(dict.fromkeys(ticker_variants))
        
        for cand in yf_candidates:
            try:
                info = yf.Ticker(cand).info
                quote_type = info.get("quoteType", "").upper()
                
                # Catch ETNs and Mutual Funds that lack "sector"
                if quote_type in ["ETF", "MUTUALFUND"]:
                    return "Investment Offices / Funds", "6726", "Finance/RealEstate"
                    
                sector = info.get("sector") or info.get("industry")
                if sector:
                    print(f"[~] {ticker}: resolved via yfinance fallback on '{cand}' -> {sector}")
                    return sector, "Unknown", _text_to_macro(sector)
            except Exception as e:
                print(f"      [X] yf.Ticker('{cand}') failed: {e}")
                continue
                
        print(f"  [!] {ticker}: yfinance returned no sector/industry info for any variant")
    except Exception as e:
        print(f"  [X] {ticker}: yfinance module error: {e}")

    return "Unknown", "Unknown", "Unknown"


# ── entry points ──────────────────────────────────────────────────────────────

if get_all == True:
    try:
        df = pd.read_feather("../Data/all_ohlcv_no_ETFs.feather")
    except Exception:
        print("Could not load DataFrame for get_all. Continuing...")
        df = pd.DataFrame({"act_symbol":[]})
        
    tickers, sectors, sic_codes, macro_sectors = [], [], [],[]
    
    for c, ticker in enumerate(df.act_symbol.unique()):
        sector, sic, macro = get_sector(ticker, MY_USER_AGENT)
        tickers.append(ticker)
        sectors.append(sector)
        sic_codes.append(sic)
        macro_sectors.append(macro)
        
        print(f"{np.round(100*c/len(df.act_symbol.unique()), 2)}% complete | {ticker} | {macro} | {sic}")
        time.sleep(0.15)

    sector_df = pd.DataFrame({
        "act_symbol": tickers, 
        "sector": sectors,
        "sic_code": sic_codes,
        "macro_sector": macro_sectors
    })
    sector_df.to_csv("../Data/sectors_0325.csv", index=False)

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
    try:
        unknowns_df = pd.read_csv("../Data/unknown_20260324.csv")
        tickers = unknowns_df["act_symbol"].tolist()
    except Exception:
        # Fallback to the exact list you provided if the CSV isn't found
        tickers =["FCRD", "FUD", "JTA", "MVC", "UAG", "USV", "UBG", "EUFX", "CROC", "CPTA", "PBY", "FCRZ", "PBB", "TPVY", "ABDC", "CPL", "DXB", "GARS", "GCE", "HDLV", "KTP", "MLPE", "OCSLL", "PIY", "SBGL", "SNH", "SNHNI", "UCI", "BALB", "BJO", "JJAB", "JJGB", "JJSB", "NAO", "SGGB", "BJJN", "COWB", "JJCB", "JJMB", "JJPB", "JJTB", "JJEB", "JJUB", "PGMB", "PBC", "OFSSI", "FCRW", "OFSSG", "LNFA", "OFSSH", "OXLCL", "OXLCM", "OXLCZ", "SSSSL", "WHFCL", "ZEXIT", "ZIEXT", "ZXIET", "PFXNZ", "OXLCI", "CBO", "MTEST", "OXLCG", "IGZ", "CBX", "OFSSO", "BANXR"]
    
    print(f"--- Testing {len(tickers)} Unknowns ---")
    results =[]
    
    for c, ticker in enumerate(tickers):
        sector, sic, macro = get_sector(ticker, MY_USER_AGENT)
        results.append({
            "act_symbol": ticker, 
            "sector": sector,
            "sic_code": sic,
            "macro_sector": macro
        })
        print(f"{np.round(100*c/len(tickers), 2)}% complete | Ticker: {ticker} | Macro: {macro} | SIC: {sic}")
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

if get_missing == True:
    #pull up sector data

    #find any data that is missing

    #fill in the unknowns
    pass