from src.ingestion.smard_client import SMARDClient


def test_filter_ids_map_to_the_right_generation_types():
    # SMARD's documented realised-generation filter IDs. A previous version used
    # 4068/4069/4070 (PV / hard coal / pumped storage) under the wind/solar names,
    # which silently trained every model on the wrong energy sources.
    assert SMARDClient.FILTERS == {
        "wind_onshore": "4067",
        "wind_offshore": "1225",
        "solar": "4068",
    }
