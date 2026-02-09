import os
import torch
import logging
from datetime import datetime
from auto_train import run_auto_pipeline

# 1. Define the assets you want to research overnight
# Format: (Symbol, Source)
ASSETS = [
    ("AAPL", "yfinance"),    # Tech Stock
    ("TSLA", "yfinance"),    # Volatile Stock
    ("BTC/USDT", "binance"), # Crypto Leader
    ("ETH/USDT", "binance"), # Altcoin Leader
    ("^SPX", "stooq"),       # S&P 500 Index
    ("NVDA", "yfinance"),    # High-Growth Tech
]

class Args:
    """Helper to mimic CLI arguments for the auto_pipeline."""
    def __init__(self, symbol, source, model_type="pc", trials=40, episodes=300):
        self.symbol = symbol
        self.source = source
        self.model_type = model_type
        self.trials = trials
        self.episodes = episodes

def run_nightly_research():
    start_time = datetime.now()
    report_path = f"overnight_report_{start_time.strftime('%Y%m%d')}.txt"
    
    # Setup Logging
    logging.basicConfig(
        filename=f"overnight_{start_time.strftime('%Y%m%d')}.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    with open(report_path, "w") as f:
        f.write(f"=== OVERNIGHT RESEARCH REPORT: {start_time.strftime('%Y-%m-%d')} ===\n\n")

    print(f"[*] Starting Overnight Research on {len(ASSETS)} assets...")
    
    for symbol, source in ASSETS:
        asset_start = datetime.now()
        print(f"\n>>> PROCESSING: {symbol} ({source})")
        
        try:
            # Configure intensive research settings
            args = Args(symbol, source, model_type="pc", trials=40, episodes=300)
            
            # Run the full pipeline
            run_auto_pipeline(args)
            
            # Log Success
            duration = datetime.now() - asset_start
            msg = f"SUCCESS: {symbol} | Duration: {duration}"
            logging.info(msg)
            
            with open(report_path, "a") as f:
                f.write(f"{msg}\n")
                
        except Exception as e:
            msg = f"FAILED: {symbol} | Error: {str(e)}"
            print(f"[!] {msg}")
            logging.error(msg)
            with open(report_path, "a") as f:
                f.write(f"{msg}\n")

    total_duration = datetime.now() - start_time
    footer = f"\n=== RESEARCH FINISHED | Total Time: {total_duration} ==="
    print(footer)
    with open(report_path, "a") as f:
        f.write(footer)

if __name__ == "__main__":
    # Optimize CPU usage
    if not torch.cuda.is_available():
        import os
        num_threads = os.cpu_count() // 2 if os.cpu_count() > 4 else os.cpu_count()
        torch.set_num_threads(num_threads)
        
    run_nightly_research()
