import requests
import pandas as pd
from typing import Optional, List
from datetime import datetime, date, timezone
from src.utils.logger import log

class SMARDClient:
    """Fallback client to fetch actual generation data from SMARD."""
    
    BASE_URL = "https://www.smard.de/app/chart_data"
    REGION_DE = "DE"
    RESOLUTION = "hour"
    
    # Filter IDs for realised generation
    FILTERS = {
        "wind_onshore": "4068",
        "wind_offshore": "4069",
        "solar": "4070"
    }

    def _get_index(self, filter_id: str) -> Optional[List[int]]:
        """Get the available timestamps (chunks) for a filter."""
        url = f"{self.BASE_URL}/{filter_id}/{self.REGION_DE}/index_{self.RESOLUTION}.json"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json().get("timestamps", [])
        except requests.RequestException as e:
            log.error(f"Failed to fetch SMARD index for filter {filter_id}: {e}")
            return None

    def _fetch_series(self, filter_id: str, timestamp: int) -> Optional[pd.DataFrame]:
        """Fetch data series for a specific timestamp chunk."""
        url = f"{self.BASE_URL}/{filter_id}/{self.REGION_DE}/{filter_id}_{self.REGION_DE}_{self.RESOLUTION}_{timestamp}.json"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json().get("series", [])
            if not data:
                return None
            df = pd.DataFrame(data, columns=["time_ms", "value"])
            # Convert ms to datetime UTC
            df['time'] = pd.to_datetime(df['time_ms'], unit='ms').dt.tz_localize('UTC')
            df.drop(columns=["time_ms"], inplace=True)
            df.set_index("time", inplace=True)
            return df
        except requests.RequestException as e:
            log.error(f"Failed to fetch SMARD data for {filter_id} at {timestamp}: {e}")
            return None

    def fetch_historical(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch wind and solar generation for the specified date range.
        Returns a DataFrame indexed by time with columns for each source.
        """
        start_ts = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc).timestamp() * 1000)
        
        log.info(f"Fetching SMARD generation data from {start_date} to {end_date}")
        
        all_dfs = []
        for name, filter_id in self.FILTERS.items():
            log.info(f"Fetching {name} data...")
            index = self._get_index(filter_id)
            if not index:
                continue
            
            # Find relevant chunks
            # A chunk might start before start_ts but contain data up to start_ts. 
            # We'll just take all chunks that are <= end_ts, and either >= start_ts or the one right before start_ts.
            relevant_chunks = [ts for ts in index if ts <= end_ts]
            # Optimization: could further narrow it down, but let's just grab the last few or all relevant, 
            # then filter the final DataFrame. To be safe, we fetch all chunks that might overlap.
            # SMARD chunks are usually weekly or yearly.
            
            source_dfs = []
            # Reverse iterate or find the chunk right before start_ts
            idx_start = 0
            for i, ts in enumerate(relevant_chunks):
                if ts <= start_ts:
                    idx_start = i
            
            for ts in relevant_chunks[idx_start:]:
                chunk_df = self._fetch_series(filter_id, ts)
                if chunk_df is not None:
                    chunk_df.rename(columns={"value": name}, inplace=True)
                    source_dfs.append(chunk_df)
            
            if source_dfs:
                combined = pd.concat(source_dfs)
                # Remove duplicates in case of overlapping chunks (shouldn't happen, but safe)
                combined = combined[~combined.index.duplicated(keep='first')]
                all_dfs.append(combined)
                
        if not all_dfs:
            log.error("No SMARD data could be fetched.")
            return None
            
        # Merge all sources
        master_df = all_dfs[0]
        for df in all_dfs[1:]:
            master_df = master_df.join(df, how="outer")
            
        # Filter strictly by the requested dates
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        # Add 1 day minus 1 min for inclusive end date
        end_dt = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        master_df = master_df[(master_df.index >= start_dt) & (master_df.index <= end_dt)]
        
        # SMARD returns MWh for quarterhour/hour. We should ensure it's in MWh.
        # Since resolution is 'hour', the values are in MWh/h which is MW.
        
        log.info(f"Successfully fetched {len(master_df)} rows from SMARD.")
        return master_df
