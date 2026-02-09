import os
import pandas as pd
from datetime import datetime

class MarketDataset:
    """Manages a batched local cache of market data to support large datasets on low RAM."""
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_path(self, symbol, year, source):
        safe_symbol = symbol.replace("/", "_").replace("^", "IDX_")
        symbol_dir = os.path.join(self.cache_dir, safe_symbol, source)
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)
        return os.path.join(symbol_dir, f"{year}.csv")

    def save_batch(self, df, symbol, source):
        """Saves data partitioned by year."""
        if df is None or df.empty:
            return
        
        # Group by year and save separate files
        df_to_save = df.copy()
        df_to_save['Year'] = df_to_save.index.year
        
        for year, group in df_to_save.groupby('Year'):
            path = self._get_path(symbol, year, source)
            # Remove helper column before saving
            save_group = group.drop(columns=['Year'])
            
            if os.path.exists(path):
                # Merge with existing data to avoid duplicates
                existing = pd.read_csv(path, index_col=0, parse_dates=True)
                combined = pd.concat([existing, save_group])
                combined = combined[~combined.index.duplicated(keep='first')].sort_index()
                combined.to_csv(path)
            else:
                save_group.to_csv(path)

    def load_full_range(self, symbol, start_date, end_date, source):
        """Attempts to load data from cache first, returns None if range is missing."""
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        
        dfs = []
        for year in range(start_year, end_year + 1):
            path = self._get_path(symbol, year, source)
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                dfs.append(df)
            else:
                # If any year in range is missing, we consider the cache incomplete
                return None
        
        full_df = pd.concat(dfs).sort_index()
        # Filter exact dates
        return full_df[(full_df.index >= start_date) & (full_df.index <= end_date)]

    def get_batches(self, symbol, source, batch_size_years=1):
        """Generator for low-RAM systems to process data in chunks."""
        symbol_dir = os.path.join(self.cache_dir, symbol.replace("/", "_"), source)
        if not os.path.exists(symbol_dir):
            return
            
        years = sorted([int(f.split('.')[0]) for f in os.listdir(symbol_dir) if f.endswith('.csv')])
        
        for i in range(0, len(years), batch_size_years):
            chunk_years = years[i : i + batch_size_years]
            dfs = [pd.read_csv(self._get_path(symbol, y, source), index_col=0, parse_dates=True) for y in chunk_years]
            yield pd.concat(dfs).sort_index()
