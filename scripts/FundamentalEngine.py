import pandas as pd
import numpy as np

'''
Everything is written to work on quarterly data
'''

#0. Functions
def compute_eps_growth_qoq(df: pd.DataFrame):
    """
    EPS growth quarter-over-quarter.
    Computes the shifted series once and reuses it to avoid
    double-groupby misalignment.
    """
    shifted = df.groupby('act_symbol')['eps'].shift(1)
    return (df['eps'] - shifted) / shifted.abs().replace(0, np.nan)

def compute_eps_growth_yoy(df: pd.DataFrame):
    """
    EPS growth year-over-year (4 quarters back).
    Assumes quarterly data — raises a warning if annual rows are detected.
    """
    shifted = df.groupby('act_symbol')['eps'].shift(4)
    return (df['eps'] - shifted) / shifted.abs().replace(0, np.nan)

def compute_revenue_growth_yoy(df: pd.DataFrame):
    """
    Revenue growth year-over-year (4 quarters back).
    Assumes quarterly data — raises a warning if annual rows are detected.
    """
    shifted = df.groupby('act_symbol')['revenue'].shift(4)
    return (df['revenue'] - shifted) / shifted.abs().replace(0, np.nan)


def compute_asset_growth_yoy(df: pd.DataFrame):
    """
    Total asset growth year-over-year (4 quarters back).
    Assumes quarterly data — raises a warning if annual rows are detected.
    """
    shifted = df.groupby('act_symbol')['total_assets'].shift(4)
    return (df['total_assets'] - shifted) / shifted.abs().replace(0, np.nan)

def compute_quick_ratio(df: pd.DataFrame):
    """
    Quick ratio using the additive method:
        (cash + short_term_investments + net_receivables) / current_liabilities

    This is more conservative than subtracting inventory from current assets,
    as it excludes illiquid current assets like prepaid expenses, deferred costs,
    and other non-cash items that the inventory-subtraction method implicitly includes.

    Required columns:
        - cash_and_equivalents
        - short_term_investments
        - net_receivables
        - current_liabilities
    """
    liquid_assets = (
        df['cash_and_equivalents']
        + df['short_term_investments'].fillna(0)
        + df['net_receivables'].fillna(0)
    )
    return liquid_assets / df['current_liabilities'].replace(0, np.nan)

# ==========================================
# 1. THE REGISTRY
# ==========================================
FUNDAMENTAL_REGISTRY = {
    
    # --- PROBABILITY & EFFICIENCY ---
    'ROE': {
        'fn': lambda df: df['net_income'] / df['total_equity'].replace(0, np.nan),
        'inputs': ['net_income', 'total_equity']
    },
    'ROA': {
        'fn': lambda df: df['net_income'] / df['total_assets'].replace(0, np.nan),
        'inputs':['net_income', 'total_assets']
    },
    'GROSS_MARGIN': {
        'fn': lambda df: df['gross_profit'] / df['revenue'].replace(0, np.nan),
        'inputs': ['gross_profit', 'revenue']
    },
    'OPERATING_MARGIN': {
        'fn': lambda df: df['operating_income'] / df['revenue'].replace(0, np.nan),
        'inputs':['operating_income', 'revenue']
    },
    'ASSET_TURNOVER': {
        'fn': lambda df: df['revenue'] / df['total_assets'].replace(0, np.nan),
        'inputs': ['revenue', 'total_assets']
    },
    'CASH_FLOW_ROA': {
        'fn': lambda df: df['operating_cash_flow'] / df['total_assets'].replace(0, np.nan),
        'inputs':['operating_cash_flow', 'total_assets']
    },

    # --- LIQUIDITY / SOLVENCY (Balance Sheet) ---
    'CURRENT_RATIO': {
        'fn': lambda df: df['current_assets'] / df['current_liabilities'].replace(0, np.nan),
        'inputs':['current_assets', 'current_liabilities']
    },
    'QUICK_RATIO': {
        'fn': lambda df: (df['current_assets'] - df['inventory']) / df['current_liabilities'].replace(0, np.nan),
        'inputs':['current_assets', 'inventory', 'current_liabilities']
    },
    'DEBT_TO_EQUITY': {
        'fn': lambda df: df['total_debt'] / df['total_equity'].replace(0, np.nan),
        'inputs': ['total_debt', 'total_equity']
    },
    'ASSET_LEVERAGE': {
        'fn': lambda df: df['total_assets'] / df['total_equity'].replace(0, np.nan),
        'inputs':['total_assets', 'total_equity']
    },
    'CASH_RATIO': {
        'fn': lambda df: df['cash_and_equivalents'] / df['current_liabilities'].replace(0, np.nan),
        'inputs':['cash_and_equivalents', 'current_liabilities']
    },

    # --- ADVANCED QUANT FACTORS ---
    'SLOANS_ACCRUALS': {
        'fn': lambda df: (df['net_income'] - df['operating_cash_flow']) / df['total_assets'].replace(0, np.nan),
        'inputs':['net_income', 'operating_cash_flow', 'total_assets']
    },
    'CF_TO_NET_INCOME': {
        'fn': lambda df: df['operating_cash_flow'] / df['net_income'].replace(0, np.nan),
        'inputs':['operating_cash_flow', 'net_income']
    },

    # --- MOMENTUM (Assumes Sparse Quarterly Rows) ---
    'EPS_GROWTH_QOQ': {
        # Shift 1 = 1 Quarter back
        'fn': compute_eps_growth_qoq,
        'inputs': ['eps']
    },
    'EPS_GROWTH_YOY': {
        # Shift 4 = 4 Quarters back (1 Year)
        'fn': compute_eps_growth_yoy,
        'inputs': ['eps']
    },
    'REVENUE_GROWTH_YOY': {
        'fn': lambda df: (df['revenue'] - df.groupby('act_symbol')['revenue'].shift(4)) / df.groupby('act_symbol')['revenue'].shift(4).abs().replace(0, np.nan),
        'inputs': ['revenue']
    },
    'ASSET_GROWTH_YOY': {
        'fn': lambda df: (df['total_assets'] - df.groupby('act_symbol')['total_assets'].shift(4)) / df.groupby('act_symbol')['total_assets'].shift(4).abs().replace(0, np.nan),
        'inputs':['total_assets']
    },

    # --- EVENT-DRIVEN ---
    'SUE': {
        # Standardized Unexpected Earnings: (Actual - Estimate) / Standard Deviation of Estimates
        'fn': lambda df: (df['eps'] - df['eps_est']) / df['eps_est_std'].replace(0, np.nan),
        'inputs': ['eps', 'eps_est', 'eps_est_std']
    },
    'PEAD_SURPRISE_PROXY': {
        # Actual drift requires price data. Here we calculate Earnings Surprise %, 
        # which is the fundamental catalyst/proxy for PEAD.
        'fn': lambda df: (df['eps'] - df['eps_est']) / df['eps_est'].abs().replace(0, np.nan),
        'inputs':['eps', 'eps_est']
    },
    'DAYS_TO_NEXT_EARNINGS': {
        'fn': lambda df: (pd.to_datetime(df['next_earnings_date']) - pd.to_datetime(df['date'])).dt.days,
        'inputs':['date', 'next_earnings_date']
    },
    'DAYS_SINCE_LAST_EARNINGS': {
        'fn': lambda df: (pd.to_datetime(df['date']) - pd.to_datetime(df['last_earnings_date'])).dt.days,
        'inputs':['date', 'last_earnings_date']
    },
    # --- DATA AVAILABILITY ---
    'HAS_FUNDAMENTALS': {
        # If a row exists in the fundamental dataframe for this stock/quarter, 
        # it inherently means fundamental data is available. We assign it a 1.
        # (clean_analysis.py will automatically fill the missing/NaN rows with 0 later).
        'fn': lambda df: np.ones(len(df)),
        'inputs': []  # No specific raw columns are required
    },
}

# ==========================================
# 2. THE REQUEST OBJECT
# ==========================================
class FundamentalRequest:
    def __init__(self, name, alias=None, transform=None, transform_params=None, neutralization='market'):
        if name not in FUNDAMENTAL_REGISTRY:
            raise ValueError(f"Fundamental Feature '{name}' not found in Registry")
        self.name = name
        
        self.transform = transform
        self.transform_params = transform_params if transform_params else {}
        self.neutralization = neutralization
        
        # 1. Base Name (The raw computed feature, e.g., F_ROE)
        self.base_alias = alias if alias else f"F_{name}"
        
        # 2. Final Name (Appends the transform suffix, perfectly matching FeatureEngine)
        self.alias = self.base_alias
        if self.transform:
            if self.transform in ['rank', 'demean', 'cs_zscore'] and self.neutralization == 'sector':
                self.alias += f"_SECTOR_{self.transform.upper()}"
            else:
                self.alias += f"_{self.transform.upper()}"

# ==========================================
# 3. THE ENGINE
# ==========================================
class FundamentalEngine:
    def __init__(self, requests):
        self.requests = requests

    def compute(self, df):
        print(f"Fundamental Engine: Computing {len(self.requests)} sparse features...")
        
        # 0. MAP DOLTHUB COLUMNS TO REGISTRY EXPECTATIONS
        # This ensures your specific CSV works perfectly with the standard registry formulas
        rename_map = {
            'sales': 'revenue',
            'diluted_net_eps': 'eps',
            'net_cash_from_operating_activities': 'operating_cash_flow',
            'inventories': 'inventory',
            'total_current_assets': 'current_assets',
            'total_current_liabilities': 'current_liabilities',
            'total_liabilities': 'total_debt', 
            'pretax_income': 'operating_income' 
        }
        df = df.rename(columns=rename_map)

        # 1. Safety Sort (Critical for Growth/Shift calculations grouping by symbol)
        df = df.sort_values(['act_symbol', 'date']).copy()
        
        # 2. Compute Base Features
        for req in self.requests:
            print(f"  -> {req.name}")
            config = FUNDAMENTAL_REGISTRY[req.name]
            
            missing = [col for col in config['inputs'] if col not in df.columns]
            
            if missing:
                print(f"[Warning] Missing raw columns {missing} for {req.name}. Yielding NaNs.")
                df[req.base_alias] = np.nan
            else:
                try:
                    df[req.base_alias] = config['fn'](df)
                    df[req.base_alias] = df[req.base_alias].replace([np.inf, -np.inf], np.nan)
                except Exception as e:
                    print(f"[Error] Failed to compute {req.name}: {e}. Yielding NaNs.")
                    df[req.base_alias] = np.nan
                    
        # ---------------------------------------------------------
        # 3. ANTI-LOOK-AHEAD BIAS LOGIC (THE 3-MONTH SHIFT)
        # ---------------------------------------------------------
        print("Fundamental Engine: Applying 3-Month Lag to prevent Look-Ahead Bias...")
        
        # Shift the reporting date forward by 3 months. 
        # The Q3 report (10-31) becomes "visible" to the model on 01-31.
        df['date'] = df['date'] + pd.DateOffset(months=3)
        
        # Force the date to snap to the exact End of the Month.
        # (e.g. 04-30 + 3 months -> 07-30. This corrects it to 07-31).
        df['date'] = df['date'] + pd.offsets.MonthEnd(0)
        
        # ---------------------------------------------------------
        # 4. APPLY TRANSFORMS AND NEUTRALIZATIONS
        # ---------------------------------------------------------
        
        # Load Sector Data if required
        self.needs_sector_data = any(
            req.transform in ['rank', 'demean', 'cs_zscore'] and getattr(req, 'neutralization', 'market') == 'sector'
            for req in self.requests
        )
        if self.needs_sector_data:
            sectors_df = pd.read_csv("../Data/sectors_info.csv", usecols=['act_symbol', 'NAICS_macro']).drop_duplicates(subset=['act_symbol'])
            df = df.merge(sectors_df, on='act_symbol', how='left')
            df['NAICS_macro'] = df['NAICS_macro'].fillna('UNKNOWN')

        # Create Temporary Quarter Column for valid Cross-Sectional Math on shifted dates
        df['_temp_quarter'] = df['date'].dt.to_period('Q')

        base_cols_used_for_transforms = set()
        
        for req in self.requests:
            if not req.transform:
                continue

            base_col = req.base_alias
            final_col = req.alias
            
            if base_col not in df.columns:
                continue

            # Determine grouping: Market (Quarter) vs Sector (Quarter + Sector)
            if req.transform in ['rank', 'demean', 'cs_zscore']:
                cs_group = ['_temp_quarter', 'NAICS_macro'] if getattr(req, 'neutralization', 'market') == 'sector' else ['_temp_quarter']
            
            # Apply identical transformations to FeatureEngine
            if req.transform == 'rank':
                df[final_col] = df.groupby(cs_group)[base_col].rank(pct=True)
                
            elif req.transform == 'demean':
                means = df.groupby(cs_group)[base_col].transform('mean')
                df[final_col] = df[base_col] - means

            elif req.transform == 'cs_zscore':
                means = df.groupby(cs_group)[base_col].transform('mean')
                stds = df.groupby(cs_group)[base_col].transform('std')
                df[final_col] = (df[base_col] - means) / stds.replace(0, np.nan)
                
            elif req.transform == 'binary':
                thresh = req.transform_params.get('threshold', 0)
                df[final_col] = np.where(df[base_col] > thresh, 1, 0)
                df.loc[df[base_col].isna(), final_col] = np.nan
                
            elif req.transform == 'regime':
                thresh = req.transform_params.get('threshold', 0.02)
                conditions = [(df[base_col] > thresh), (df[base_col] < -thresh)]
                choices = [1, -1]
                df[final_col] = np.select(conditions, choices, default=0)
                df.loc[df[base_col].isna(), final_col] = np.nan

            base_cols_used_for_transforms.add(base_col)

        # 5. Cleanup Memory and Intermediate Columns
        df.drop(columns=['_temp_quarter'], inplace=True)

        final_requested_cols = set([req.alias for req in self.requests])
        cols_to_drop = [c for c in base_cols_used_for_transforms if c not in final_requested_cols]
        
        if getattr(self, 'needs_sector_data', False) and 'NAICS_macro' in df.columns:
            cols_to_drop.append('NAICS_macro')
            
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        return df