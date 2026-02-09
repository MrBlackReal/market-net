import torch
import overnight_research
from auto_train import run_auto_pipeline
import logging
from datetime import datetime, timedelta
from src.data import export_processed_data

# Override the ASSETS list for a quick test
overnight_research.ASSETS = [("AAPL", "yfinance")]

def run_micro_test():
    print("🧪 Starting Micro-Test for Overnight Research...")
    
    for symbol, source in overnight_research.ASSETS:
        print(f"\n>>> TESTING EXPORT: {symbol}")
        today = datetime(2026, 2, 8)
        start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        export_processed_data(symbol, start_date, end_date, source=source)

        print(f"\n>>> TESTING PIPELINE: {symbol}")
        class SmallArgs:
            def __init__(self, symbol, source):
                self.symbol = symbol
                self.source = source
                self.model_type = "pc"
                self.trials = 1
                self.episodes = 1
        
        args = SmallArgs(symbol, source)
        run_auto_pipeline(args)
        
    print("\n=== TEST FINISHED ===")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        import os
        num_threads = os.cpu_count() // 2 if os.cpu_count() > 4 else os.cpu_count()
        torch.set_num_threads(num_threads)
        
    run_micro_test()