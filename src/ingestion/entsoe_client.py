import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from typing import Optional
import pandas as pd
import requests
from src.utils.logger import log

# ENTSO-E Transparency Platform: Actual Generation per Type.
# https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
NS = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationdocument:4:0"}


class ENTSOEClient:
    """Client for the ENTSO-E Transparency Platform (Actual Generation per Type, A75/A16).

    NOTE: this implementation follows ENTSO-E's documented API (endpoint, document/process
    types, PSR codes, XML schema) but has not been exercised against a live API key — none
    was available while writing it. fetch_historical() returns None on any failure (bad key,
    network error, unparseable/error response, no data), which the ingestion pipeline treats
    as "fall back to SMARD" — the same behavior as before, just now actually attempting a
    real call instead of unconditionally stubbing it out. If you have a real ENTSOE_API_KEY,
    this is the code path to verify first.
    """

    BASE_URL = "https://web-api.tp.entsoe.eu/api"
    DOMAIN_DE_LU = "10Y1001A1001A82H"  # Germany-Luxembourg bidding zone (EIC code)
    PSR_TYPES = {
        "wind_onshore": "B19",
        "wind_offshore": "B18",
        "solar": "B16",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_historical(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        if not self.api_key or self.api_key == "your_key_here":
            log.warning("No valid ENTSO-E API key provided.")
            return None

        series = []
        for name, psr_type in self.PSR_TYPES.items():
            df = self._fetch_series(name, psr_type, start_date, end_date)
            if df is None:
                log.warning(f"ENTSO-E: failed to fetch {name}; aborting so the pipeline falls back to SMARD.")
                return None
            series.append(df)

        combined = series[0]
        for s in series[1:]:
            combined = combined.join(s, how="outer")
        log.info(f"ENTSO-E: fetched {len(combined)} rows for {list(self.PSR_TYPES)}.")
        return combined

    def _fetch_series(self, name: str, psr_type: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        params = {
            "securityToken": self.api_key,
            "documentType": "A75",  # Actual generation per type
            "processType": "A16",  # Realised
            "in_Domain": self.DOMAIN_DE_LU,
            "psrType": psr_type,
            "periodStart": self._fmt(start_date),
            "periodEnd": self._fmt(end_date, end_of_day=True),
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            log.error(f"ENTSO-E request failed for {name}: {e}")
            return None

        try:
            return self._parse_generation_xml(response.text, name)
        except ET.ParseError as e:
            log.error(f"ENTSO-E: could not parse response for {name}: {e}")
            return None

    @staticmethod
    def _parse_generation_xml(xml_text: str, column_name: str) -> Optional[pd.DataFrame]:
        """Parses a GL_MarketDocument response into an hourly-indexed single-column DataFrame.
        Returns None if the document has no TimeSeries (e.g. an ENTSO-E error/acknowledgement
        document for a bad key or empty period — still valid XML, just a different schema)."""
        root = ET.fromstring(xml_text)
        rows = []
        for timeseries in root.findall("ns:TimeSeries", NS):
            period = timeseries.find("ns:Period", NS)
            if period is None:
                continue
            start_el = period.find("ns:timeInterval/ns:start", NS)
            resolution_el = period.find("ns:resolution", NS)
            if start_el is None or resolution_el is None:
                continue

            period_start = datetime.strptime(start_el.text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            step = pd.Timedelta(minutes=15) if resolution_el.text == "PT15M" else pd.Timedelta(minutes=60)

            for point in period.findall("ns:Point", NS):
                position_el = point.find("ns:position", NS)
                quantity_el = point.find("ns:quantity", NS)
                if position_el is None or quantity_el is None:
                    continue
                timestamp = period_start + (int(position_el.text) - 1) * step
                rows.append((timestamp, float(quantity_el.text)))

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["time", column_name]).set_index("time")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        # ENTSO-E can report sub-hourly (PT15M) resolution; resample to hourly to match
        # SMARD/weather granularity that the rest of the pipeline expects.
        return df.resample("h").mean()

    @staticmethod
    def _fmt(d: date, end_of_day: bool = False) -> str:
        dt = datetime.combine(d, datetime.min.time())
        if end_of_day:
            dt = dt + pd.Timedelta(days=1)
        return dt.strftime("%Y%m%d%H%M")
