'''
Adds NAICS_macro column to sectors_info.csv by mapping 4-digit SIC codes
to the 20 broad 2-digit NAICS sectors.

Usage:
    python add_naics_sectors.py
'''
import pandas as pd

# ====================================================================
# MAPPING
# ====================================================================
SIC_OVERRIDES = {
    # Publishing → Information (SIC calls this Manufacturing)
    271: 'Information', 272: 'Information', 273: 'Information', 274: 'Information',

    # Transportation (SIC 40-47)
    401: 'Transportation & Warehousing', 410: 'Transportation & Warehousing',
    411: 'Transportation & Warehousing', 412: 'Transportation & Warehousing',
    413: 'Transportation & Warehousing', 414: 'Transportation & Warehousing',
    415: 'Transportation & Warehousing', 421: 'Transportation & Warehousing',
    422: 'Transportation & Warehousing', 423: 'Transportation & Warehousing',
    440: 'Transportation & Warehousing', 441: 'Transportation & Warehousing',
    442: 'Transportation & Warehousing', 443: 'Transportation & Warehousing',
    444: 'Transportation & Warehousing', 448: 'Transportation & Warehousing',
    449: 'Transportation & Warehousing', 450: 'Transportation & Warehousing',
    451: 'Transportation & Warehousing', 452: 'Transportation & Warehousing',
    458: 'Transportation & Warehousing', 461: 'Transportation & Warehousing',
    470: 'Transportation & Warehousing', 471: 'Transportation & Warehousing',
    472: 'Transportation & Warehousing', 473: 'Transportation & Warehousing',
    474: 'Transportation & Warehousing', 478: 'Transportation & Warehousing',

    # Communications → Information (SIC calls this Transport/Comm)
    481: 'Information', 482: 'Information', 483: 'Information',
    484: 'Information', 489: 'Information',

    # Utilities (SIC calls this Transport/Comm)
    491: 'Utilities', 492: 'Utilities', 493: 'Utilities', 494: 'Utilities',
    495: 'Utilities', 496: 'Utilities', 497: 'Utilities',

    # Finance & Insurance (SIC 60-64, 67)
    600: 'Finance & Insurance', 601: 'Finance & Insurance', 602: 'Finance & Insurance',
    603: 'Finance & Insurance', 606: 'Finance & Insurance', 608: 'Finance & Insurance',
    609: 'Finance & Insurance', 610: 'Finance & Insurance', 611: 'Finance & Insurance',
    614: 'Finance & Insurance', 615: 'Finance & Insurance', 616: 'Finance & Insurance',
    620: 'Finance & Insurance', 621: 'Finance & Insurance', 622: 'Finance & Insurance',
    623: 'Finance & Insurance', 628: 'Finance & Insurance',
    630: 'Finance & Insurance', 631: 'Finance & Insurance', 632: 'Finance & Insurance',
    633: 'Finance & Insurance', 635: 'Finance & Insurance', 636: 'Finance & Insurance',
    637: 'Finance & Insurance', 639: 'Finance & Insurance',
    640: 'Finance & Insurance', 641: 'Finance & Insurance',
    670: 'Finance & Insurance', 671: 'Finance & Insurance', 672: 'Finance & Insurance',
    673: 'Finance & Insurance', 677: 'Finance & Insurance', 679: 'Finance & Insurance',

    # Real Estate (SIC 65)
    650: 'Real Estate', 651: 'Real Estate', 652: 'Real Estate',
    653: 'Real Estate', 654: 'Real Estate', 655: 'Real Estate',
    6798: 'Real Estate',  # REITs — 4-digit override, checked before 679

    # Eating & drinking places → Accommodation & Food Services (NOT Retail)
    581: 'Accommodation & Food Services',
    5810: 'Accommodation & Food Services',
    5812: 'Accommodation & Food Services',
    5813: 'Accommodation & Food Services',

    # Hotels → Accommodation & Food Services
    700: 'Accommodation & Food Services', 701: 'Accommodation & Food Services',
    702: 'Accommodation & Food Services', 703: 'Accommodation & Food Services',
    704: 'Accommodation & Food Services',

    # Personal services → Other Services
    720: 'Other Services', 721: 'Other Services', 723: 'Other Services',
    724: 'Other Services', 725: 'Other Services', 726: 'Other Services',
    729: 'Other Services',

    # Business Services — the big SIC/NAICS split
    731: 'Administrative & Support', 732: 'Administrative & Support',
    733: 'Administrative & Support', 734: 'Administrative & Support',
    735: 'Administrative & Support', 736: 'Administrative & Support',
    737: 'Information', 738: 'Administrative & Support',
    7371: 'Information', 7372: 'Information', 7373: 'Information',
    7374: 'Information', 7375: 'Information', 7376: 'Information',
    7377: 'Information', 7378: 'Information', 7379: 'Information',

    # Auto/misc repair
    750: 'Other Services', 751: 'Other Services', 752: 'Other Services',
    753: 'Other Services', 754: 'Other Services',
    760: 'Other Services', 762: 'Other Services', 764: 'Other Services',
    769: 'Other Services',

    # Motion pictures → Information
    780: 'Information', 781: 'Information', 782: 'Information',
    783: 'Information', 784: 'Information',

    # Amusement & recreation
    790: 'Arts, Entertainment & Recreation', 791: 'Arts, Entertainment & Recreation',
    792: 'Arts, Entertainment & Recreation', 793: 'Arts, Entertainment & Recreation',
    794: 'Arts, Entertainment & Recreation', 799: 'Arts, Entertainment & Recreation',

    # Health services
    800: 'Health Care', 801: 'Health Care', 802: 'Health Care',
    803: 'Health Care', 804: 'Health Care', 805: 'Health Care',
    806: 'Health Care', 807: 'Health Care', 808: 'Health Care', 809: 'Health Care',

    # Legal
    810: 'Professional & Technical', 811: 'Professional & Technical',

    # Education
    820: 'Educational Services', 821: 'Educational Services',
    822: 'Educational Services', 823: 'Educational Services',
    824: 'Educational Services', 829: 'Educational Services',

    # Social services → Health Care
    830: 'Health Care', 832: 'Health Care', 833: 'Health Care',
    835: 'Health Care', 836: 'Health Care',

    # Museums
    840: 'Arts, Entertainment & Recreation', 841: 'Arts, Entertainment & Recreation',
    842: 'Arts, Entertainment & Recreation',

    # Membership orgs
    860: 'Other Services', 861: 'Other Services', 862: 'Other Services',
    863: 'Other Services', 864: 'Other Services', 865: 'Other Services',
    866: 'Other Services', 869: 'Other Services',

    # Engineering, accounting, research
    870: 'Professional & Technical', 871: 'Professional & Technical',
    872: 'Professional & Technical', 873: 'Professional & Technical',
    874: 'Management of Companies',
    890: 'Professional & Technical',
}

SIC_2DIGIT_MAP = {
    range(1, 10):   'Agriculture',
    range(10, 15):  'Mining',
    range(15, 18):  'Construction',
    range(20, 40):  'Manufacturing',
    range(40, 50):  'Transportation & Warehousing',
    range(50, 52):  'Wholesale Trade',
    range(52, 60):  'Retail Trade',
    range(60, 68):  'Finance & Insurance',
    range(70, 90):  'Other Services',
    range(91, 100): 'Public Administration',
}

def map_sic_to_naics(sic_str):
    try:
        sic = int(sic_str)
    except (ValueError, TypeError):
        return 'Unknown'
    # 4-digit exact match
    if sic in SIC_OVERRIDES:
        return SIC_OVERRIDES[sic]
    # 3-digit match
    sic_3 = sic // 10
    if sic_3 in SIC_OVERRIDES:
        return SIC_OVERRIDES[sic_3]
    # 2-digit fallback
    sic_2 = sic // 100
    for sic_range, sector in SIC_2DIGIT_MAP.items():
        if sic_2 in sic_range:
            return sector
    return 'Unknown'

# ====================================================================
# APPLY AND SAVE
# ====================================================================
if __name__ == '__main__':
    PATH = "../Data/sectors_info.csv"
    
    df = pd.read_csv(PATH)
    df['NAICS_macro'] = df['sic_code'].apply(map_sic_to_naics)
    df.to_csv(PATH, index=False)
    
    print(f"Added NAICS_macro to {len(df)} rows")
    print(f"\nDistribution:")
    print(df['NAICS_macro'].value_counts().to_string())
    print(f"\nUnknown: {(df['NAICS_macro'] == 'Unknown').sum()}")