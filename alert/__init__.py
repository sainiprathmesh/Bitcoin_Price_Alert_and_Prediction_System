import json
import time

import requests


def get_bitcoin_price():
    URL = "https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD,JPY,EUR,INR"  # REPLACE WITH CORRECT URL
    response = requests.request("GET", URL)
    response = json.loads(response.text)
    current_price = response["USD"]
    return current_price


while True:
    print(get_bitcoin_price(), time.ctime())
    time.sleep(297)
