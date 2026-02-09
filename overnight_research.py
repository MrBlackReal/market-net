import os
import torch
import multiprocessing
from datetime import datetime, timedelta
from auto_train import run_auto_pipeline
from src.data import export_processed_data

# Assets to research
ASSETS = [
    ("AAPL", "yfinance"),
    ("TSLA", "yfinance"),
    ("BTC/USDT", "binance"),
    ("ETH/USDT", "binance"),
    ("^SPX", "stooq"),
    ("NVDA", "yfinance"),
]

class Args:
    def __init__(self, symbol, source, model_type="pc", trials=40, episodes=300):
        self.symbol = symbol
        self.source = source
        self.model_type = model_type
        self.trials = trials
        self.episodes = episodes

def worker_wrapper(asset_info, threads_per_worker):
    symbol, source = asset_info
    asset_start = datetime.now()
    
    # Lock this process to its assigned threads
    torch.set_num_threads(threads_per_worker)
    
    try:
        print(f"[WORKER] Exporting Data: {symbol}")
        today = datetime(2026, 2, 8)
        start_date = (today - timedelta(days=365*10)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        export_processed_data(symbol, start_date, end_date, source=source)

        print(f"[WORKER] Starting ML Pipeline: {symbol}")
        args = Args(symbol, source, model_type="pc", trials=40, episodes=300)
        run_auto_pipeline(args)
        
        duration = datetime.now() - asset_start
        return f"SUCCESS: {symbol} | Duration: {duration}"
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return f"FAILED: {symbol} | Error: {str(e)}\n{error_msg}"

def run_nightly_research():
    start_time = datetime.now()
    report_path = f"overnight_report_{start_time.strftime('%Y%m%d')}.txt"
    
    # 1. Hardware Detection
    cpu_cores = os.cpu_count() or 4
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        num_workers = min(4, cpu_cores) 
        threads_per_worker = max(1, cpu_cores // num_workers)
        mode = f"GPU Acceleration (Workers: {num_workers})"
    else:
        num_workers = cpu_cores
        threads_per_worker = 1
        mode = f"CPU Parallel (Workers: {num_workers})"

    print(f"[*] Starting Overnight Research on {len(ASSETS)} assets...")
    print(f"[*] Hardware Mode: {mode} | Threads/Worker: {threads_per_worker}")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        task_args = [(asset, threads_per_worker) for asset in ASSETS]
        results = pool.starmap(worker_wrapper, task_args)

    total_duration = datetime.now() - start_time
    
    with open(report_path, "w") as f:
        f.write(f"=== OVERNIGHT RESEARCH REPORT: {start_time.strftime('%Y-%m-%d')} ===\n")
        f.write(f"Total Duration: {total_duration}\n")
        f.write(f"Mode: {mode}\n\n")
        for res in results:
            f.write(f"{res}\n")
            print(res)

    print(f"\n=== RESEARCH FINISHED | Total Time: {total_duration} ===")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_nightly_research()
