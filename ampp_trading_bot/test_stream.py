import asyncio
import os
from dotenv import load_dotenv
from alpaca.data.live import OptionDataStream

load_dotenv()

async def test_stream():
    stream = OptionDataStream(
        os.environ.get("ALPACA_API_KEY", ""),
        os.environ.get("ALPACA_SECRET_KEY", "")
    )
    
    async def on_trade(trade):
        print("Trade:", trade)
        
    print("Connecting for SPY...")
    stream.subscribe_trades(on_trade, "SPY")
    await stream._run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(test_stream())
    except KeyboardInterrupt:
        pass
