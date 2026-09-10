import pandas as pd
import pytest
import requests

from src.ingestion.entsoe_client import ENTSOEClient

# A schema-accurate GL_MarketDocument fixture (ENTSO-E's documented "Actual Generation
# per Type" response format) built by hand from the public API guide, since no live
# ENTSO-E API key was available to capture a real response. This verifies the parsing
# logic against the documented schema — it does not verify the live HTTP call (URL,
# query params, or that the real server still returns exactly this shape).
SAMPLE_HOURLY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationdocument:4:0">
  <mRID>sample</mRID>
  <TimeSeries>
    <mRID>1</mRID>
    <Period>
      <timeInterval>
        <start>2024-01-01T00:00Z</start>
        <end>2024-01-01T03:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>100.5</quantity></Point>
      <Point><position>2</position><quantity>110.25</quantity></Point>
      <Point><position>3</position><quantity>95.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

SAMPLE_QUARTER_HOURLY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationdocument:4:0">
  <mRID>sample</mRID>
  <TimeSeries>
    <mRID>1</mRID>
    <Period>
      <timeInterval>
        <start>2024-01-01T00:00Z</start>
        <end>2024-01-01T01:00Z</end>
      </timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>10.0</quantity></Point>
      <Point><position>2</position><quantity>20.0</quantity></Point>
      <Point><position>3</position><quantity>30.0</quantity></Point>
      <Point><position>4</position><quantity>40.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

# ENTSO-E returns a different document type (still valid XML) for errors, e.g. an
# invalid security token or an empty period — no TimeSeries element at all.
SAMPLE_ACKNOWLEDGEMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:acknowledgementdocument:4:0">
  <mRID>ack</mRID>
  <Reason>
    <code>999</code>
    <text>No matching data found</text>
  </Reason>
</Acknowledgement_MarketDocument>"""


def test_parses_hourly_generation_xml():
    df = ENTSOEClient._parse_generation_xml(SAMPLE_HOURLY_XML, "wind_onshore")
    assert list(df.columns) == ["wind_onshore"]
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert df.iloc[0]["wind_onshore"] == 100.5
    assert df.iloc[2]["wind_onshore"] == 95.0


def test_resamples_quarter_hourly_to_hourly():
    df = ENTSOEClient._parse_generation_xml(SAMPLE_QUARTER_HOURLY_XML, "solar")
    # 4 quarter-hour points starting at 00:00 fall entirely within the 00:00 hour.
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert df.iloc[0]["solar"] == pytest.approx(25.0)  # mean(10, 20, 30, 40)


def test_returns_none_for_error_document():
    assert ENTSOEClient._parse_generation_xml(SAMPLE_ACKNOWLEDGEMENT_XML, "solar") is None


def test_fetch_historical_returns_none_without_api_key():
    client = ENTSOEClient(api_key="")
    assert client.fetch_historical(pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()) is None

    client = ENTSOEClient(api_key="your_key_here")
    assert client.fetch_historical(pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()) is None


def test_fetch_historical_returns_none_on_request_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.RequestException("network error")

    monkeypatch.setattr(requests, "get", fail)
    client = ENTSOEClient(api_key="fake-key")
    assert client.fetch_historical(pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()) is None
