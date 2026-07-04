import requests

def try_url(url):
    print(url, requests.get(url).status_code)

try_url("https://www.smard.de/app/chart_data/1225/DE/index_hour.json")
try_url("https://www.smard.de/app/chart_data/1223/DE/index_hour.json")
try_url("https://www.smard.de/app/chart_data/4068/DE/index_hour.json")
try_url("https://www.smard.de/app/chart_data/1225/10000/index_hour.json")
try_url("https://www.smard.de/app/chart_data/1223/10000/index_hour.json")
try_url("https://www.smard.de/app/chart_data/1224/10000/index_hour.json")
