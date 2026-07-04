import requests
import json

url = "https://www.smard.de/app/chart_data/4068/10000/index_hour.json"
resp = requests.get(url)
print("Wind onshore index status:", resp.status_code)
if resp.status_code == 200:
    index = resp.json()["timestamps"]
    print("First few timestamps:", index[:3])
    data_url = f"https://www.smard.de/app/chart_data/4068/10000/4068_10000_hour_{index[0]}.json"
    data_resp = requests.get(data_url)
    if data_resp.status_code == 200:
         print("Data shape:", len(data_resp.json()["series"]))
         print("Sample:", data_resp.json()["series"][:3])
