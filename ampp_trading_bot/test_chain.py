import os
from datetime import date
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

load_dotenv()

client = TradingClient(
    os.environ.get("ALPACA_API_KEY", ""),
    os.environ.get("ALPACA_SECRET_KEY", ""),
    paper=True
)

req = GetOptionContractsRequest(
    underlying_symbols=["SPY"], 
    status="active",
    expiration_date_gte=date.today(),
    expiration_date_lte=date.today()
)

try:
    contracts = client.get_option_contracts(req)
    print(f"Fetched {len(contracts.option_contracts)} 0DTE contracts")
except Exception as e:
    print("Error:", e)
