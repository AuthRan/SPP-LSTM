# Nifty 50 - India's benchmark stock market index
# Source: National Stock Exchange of India (NSE)

NIFTY_50_TICKERS = [
    "ADANIENT.NS",      # Adani Enterprises
    "ADANIPORTS.NS",    # Adani Ports
    "APOLLOHOSP.NS",    # Apollo Hospitals
    "ASIANPAINT.NS",    # Asian Paints
    "AXISBANK.NS",      # Axis Bank
    "BAJAJ-AUTO.NS",    # Bajaj Auto
    "BAJAJFINSV.NS",    # Bajaj Finserv
    "BPCL.NS",          # Bharat Petroleum
    "BHARTIARTL.NS",    # Bharti Airtel
    "BRITANNIA.NS",     # Britannia Industries
    "CIPLA.NS",         # Cipla
    "COALINDIA.NS",     # Coal India
    "DIVISLAB.NS",      # Divi's Laboratories
    "DRREDDY.NS",       # Dr. Reddy's Laboratories
    "EICHERMOT.NS",     # Eicher Motors
    "GRASIM.NS",        # Grasim Industries
    "HCLTECH.NS",       # HCL Technologies
    "HDFCBANK.NS",      # HDFC Bank
    "HDFCLIFE.NS",      # HDFC Life Insurance
    "HEROMOTOCO.NS",    # Hero MotoCorp
    "HINDALCO.NS",      # Hindalco Industries
    "HINDUNILVR.NS",    # Hindustan Unilever
    "ICICIBANK.NS",     # ICICI Bank
    "ITC.NS",           # ITC Limited
    "INDUSINDBK.NS",    # IndusInd Bank
    "INFY.NS",          # Infosys
    "JSWSTEEL.NS",      # JSW Steel
    "KOTAKBANK.NS",     # Kotak Mahindra Bank
    "LT.NS",            # Larsen & Toubro
    "M&M.NS",           # Mahindra & Mahindra
    "MARUTI.NS",        # Maruti Suzuki
    "NESTLEIND.NS",     # Nestle India
    "NTPC.NS",          # NTPC Limited
    "ONGC.NS",          # Oil and Natural Gas Corp
    "POWERGRID.NS",     # Power Grid Corporation
    "RELIANCE.NS",      # Reliance Industries
    "SBILIFE.NS",       # SBI Life Insurance
    "SHRIRAMFIN.NS",    # Shriram Finance
    "SBIN.NS",          # State Bank of India
    "SUNPHARMA.NS",     # Sun Pharma
    "TCS.NS",           # Tata Consultancy Services
    "TATACONSUM.NS",    # Tata Consumer Products
    "TATAMOTORS.NS",    # Tata Motors
    "TATASTEEL.NS",     # Tata Steel
    "TECHM.NS",         # Tech Mahindra
    "TITAN.NS",         # Titan Company
    "UPL.NS",           # UPL Limited
    "ULTRACEMCO.NS",    # UltraTech Cement
    "WIPRO.NS",         # Wipro
]

def get_nifty_50_tickers():
    """Return list of Nifty 50 ticker symbols."""
    return NIFTY_50_TICKERS

def get_ticker_name(symbol):
    """Get human-readable company name from ticker."""
    ticker_names = {
        "ADANIENT.NS": "Adani Enterprises",
        "ADANIPORTS.NS": "Adani Ports",
        "APOLLOHOSP.NS": "Apollo Hospitals",
        "ASIANPAINT.NS": "Asian Paints",
        "AXISBANK.NS": "Axis Bank",
        "BAJAJ-AUTO.NS": "Bajaj Auto",
        "BAJAJFINSV.NS": "Bajaj Finserv",
        "BPCL.NS": "Bharat Petroleum",
        "BHARTIARTL.NS": "Bharti Airtel",
        "BRITANNIA.NS": "Britannia Industries",
        "CIPLA.NS": "Cipla",
        "COALINDIA.NS": "Coal India",
        "DIVISLAB.NS": "Divi's Laboratories",
        "DRREDDY.NS": "Dr. Reddy's Laboratories",
        "EICHERMOT.NS": "Eicher Motors",
        "GRASIM.NS": "Grasim Industries",
        "HCLTECH.NS": "HCL Technologies",
        "HDFCBANK.NS": "HDFC Bank",
        "HDFCLIFE.NS": "HDFC Life Insurance",
        "HEROMOTOCO.NS": "Hero MotoCorp",
        "HINDALCO.NS": "Hindalco Industries",
        "HINDUNILVR.NS": "Hindustan Unilever",
        "ICICIBANK.NS": "ICICI Bank",
        "ITC.NS": "ITC Limited",
        "INDUSINDBK.NS": "IndusInd Bank",
        "INFY.NS": "Infosys",
        "JSWSTEEL.NS": "JSW Steel",
        "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "LT.NS": "Larsen & Toubro",
        "M&M.NS": "Mahindra & Mahindra",
        "MARUTI.NS": "Maruti Suzuki",
        "NESTLEIND.NS": "Nestle India",
        "NTPC.NS": "NTPC Limited",
        "ONGC.NS": "Oil and Natural Gas Corp",
        "POWERGRID.NS": "Power Grid Corporation",
        "RELIANCE.NS": "Reliance Industries",
        "SBILIFE.NS": "SBI Life Insurance",
        "SHRIRAMFIN.NS": "Shriram Finance",
        "SBIN.NS": "State Bank of India",
        "SUNPHARMA.NS": "Sun Pharma",
        "TCS.NS": "Tata Consultancy Services",
        "TATACONSUM.NS": "Tata Consumer Products",
        "TATAMOTORS.NS": "Tata Motors",
        "TATASTEEL.NS": "Tata Steel",
        "TECHM.NS": "Tech Mahindra",
        "TITAN.NS": "Titan Company",
        "UPL.NS": "UPL Limited",
        "ULTRACEMCO.NS": "UltraTech Cement",
        "WIPRO.NS": "Wipro",
    }
    return ticker_names.get(symbol, symbol)
