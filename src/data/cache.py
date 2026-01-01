"""
Data Cache Module - src/data/cache.py

Handles caching of downloaded data to avoid redundant API calls.
Supports file-based caching with automatic expiration.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCache:
    """
    Cache financial data locally to improve performance.
    
    Features:
    - File-based caching using pickle
    - Automatic cache expiration
    - CSV backup support
    - Cache statistics
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        expiry_hours: int = 24
    ):
        """
        Initialize DataCache.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_hours: Cache expiration time in hours (default 24)
        """
        self.cache_dir = Path(cache_dir)
        self.expiry_hours = expiry_hours
        self.expiry_seconds = expiry_hours * 3600
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataCache initialized at {self.cache_dir} with {expiry_hours}h expiry")
    
    def _get_cache_path(self, ticker: str, filetype: str = "pkl") -> Path:
        """
        Get cache file path for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            filetype: File type ('pkl' or 'csv')
        
        Returns:
            Path object for cache file
        """
        # Sanitize ticker name for filename
        safe_ticker = ticker.replace(".", "_").replace("^", "")
        filename = f"{safe_ticker}_cache.{filetype}"
        return self.cache_dir / filename
    
    def is_cache_valid(self, ticker: str) -> bool:
        """
        Check if cache for ticker is valid (not expired).
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            True if cache exists and is valid
        """
        cache_path = self._get_cache_path(ticker)
        
        if not cache_path.exists():
            logger.debug(f"Cache not found for {ticker}")
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        is_valid = file_age.total_seconds() < self.expiry_seconds
        
        if is_valid:
            logger.info(f"Cache valid for {ticker} (age: {file_age.total_seconds()/3600:.1f}h)")
        else:
            logger.info(f"Cache expired for {ticker} (age: {file_age.total_seconds()/3600:.1f}h)")
        
        return is_valid
    
    def save_to_cache(
        self,
        ticker: str,
        data: pd.DataFrame,
        save_csv: bool = False
    ) -> bool:
        """
        Save DataFrame to cache.
        
        Args:
            ticker: Stock ticker symbol
            data: DataFrame to cache
            save_csv: Also save as CSV (default False)
        
        Returns:
            True if successful
        """
        try:
            cache_path = self._get_cache_path(ticker, "pkl")
            
            # Save as pickle (faster)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Cached {len(data)} records for {ticker} at {cache_path}")
            
            # Optionally save as CSV
            if save_csv:
                csv_path = self._get_cache_path(ticker, "csv")
                data.to_csv(csv_path)
                logger.info(f"Also saved CSV backup to {csv_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving cache for {ticker}: {str(e)}")
            return False
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from cache.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame if found and valid, else None
        """
        if not self.is_cache_valid(ticker):
            return None
        
        try:
            cache_path = self._get_cache_path(ticker, "pkl")
            
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded {len(data)} records for {ticker} from cache")
            
            return data
        
        except Exception as e:
            logger.error(f"Error loading cache for {ticker}: {str(e)}")
            return None
    
    def clear_cache(self, ticker: Optional[str] = None) -> bool:
        """
        Clear cache files.
        
        Args:
            ticker: Specific ticker to clear (None = all)
        
        Returns:
            True if successful
        """
        try:
            if ticker is None:
                # Clear all cache files
                for file in self.cache_dir.glob("*_cache.*"):
                    file.unlink()
                logger.info("Cleared all cache files")
            else:
                # Clear specific ticker
                pkl_path = self._get_cache_path(ticker, "pkl")
                csv_path = self._get_cache_path(ticker, "csv")
                
                if pkl_path.exists():
                    pkl_path.unlink()
                if csv_path.exists():
                    csv_path.unlink()
                
                logger.info(f"Cleared cache for {ticker}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_info(self) -> dict:
        """
        Get information about cached files.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*_cache.*"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        info = {
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "expiry_hours": self.expiry_hours,
            "files": {}
        }
        
        # Add info for each file
        for file in cache_files:
            file_age = datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)
            info["files"][file.name] = {
                "size_kb": file.stat().st_size / 1024,
                "age_hours": file_age.total_seconds() / 3600,
                "valid": file_age.total_seconds() < self.expiry_seconds
            }
        
        return info
    
    def display_cache_info(self) -> None:
        """Print cache information to console."""
        info = self.get_cache_info()
        
        print("\n" + "="*60)
        print("CACHE INFORMATION")
        print("="*60)
        print(f"Cache Directory: {info['cache_dir']}")
        print(f"Total Files: {info['num_files']}")
        print(f"Total Size: {info['total_size_mb']:.2f} MB")
        print(f"Expiry Time: {info['expiry_hours']} hours")
        print("\nCached Files:")
        
        for filename, details in info["files"].items():
            status = "✅ VALID" if details["valid"] else "❌ EXPIRED"
            print(f"  {filename:30s} | {details['size_kb']:8.2f} KB | {details['age_hours']:6.1f}h | {status}")
        
        print("="*60 + "\n")


class CachedDataLoader:
    """
    Wrapper combining DataLoader with caching.
    Automatically uses cache when available.
    """
    
    def __init__(self, loader, cache: Optional[DataCache] = None):
        """
        Initialize CachedDataLoader.
        
        Args:
            loader: DataLoader instance
            cache: DataCache instance (creates new if None)
        """
        self.loader = loader
        self.cache = cache or DataCache()
    
    def fetch_with_cache(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch data with cache support.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            force_refresh: Ignore cache and fetch fresh data
        
        Returns:
            DataFrame with market data
        """
        # Try cache first
        if not force_refresh:
            cached_data = self.cache.load_from_cache(ticker)
            if cached_data is not None:
                logger.info(f"Using cached data for {ticker}")
                return cached_data
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {ticker}")
        data = self.loader.fetch_from_yfinance(ticker, start_date, end_date)
        
        # Cache the data
        self.cache.save_to_cache(ticker, data)
        
        return data


# Example usage
if __name__ == "__main__":
    from loader import DataLoader
    
    # Initialize cache
    cache = DataCache(cache_dir="data/cache", expiry_hours=24)
    
    # Initialize loader
    loader = DataLoader()
    
    # Create cached loader
    cached_loader = CachedDataLoader(loader, cache)
    
    # First fetch (from API)
    print("First fetch (from API):")
    df1 = cached_loader.fetch_with_cache("^NSEI", start_date="2023-01-01")
    print(f"Fetched {len(df1)} records\n")
    
    # Second fetch (from cache)
    print("Second fetch (from cache):")
    df2 = cached_loader.fetch_with_cache("^NSEI", start_date="2023-01-01")
    print(f"Fetched {len(df2)} records\n")
    
    # Show cache info
    cache.display_cache_info()
    
    # Check if data is same
    print(f"Data identical: {df1.equals(df2)}")

