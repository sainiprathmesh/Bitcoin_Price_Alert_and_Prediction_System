import json
import time

import requests
from boltiot import Bolt, Sms, Email

from alert import conf


def send_telegram_message(message):
    url = "https://api.telegram.org/" + conf.telegram_bot_id + "/sendMessage"
    data = {"chat_id": conf.telegram_chat_id,
            "text": message
            }
    try:
        response = requests.request(
            "GET",
            url,
            params=data
        )
        print("This is the Telegram response")
        print(response.text)
        telegram_data = json.loads(response.text)
        return telegram_data["ok"]
    except Exception as e:
        print("An error occurred in sending the alert message via Telegram")
        print(e)
        return False


def get_bitcoin_price():
    URL = "https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD,JPY,EUR,INR"  # REPLACE WITH CORRECT URL
    respons = requests.request("GET", URL)
    respons = json.loads(respons.text)
    current_price = respons["USD"]
    return current_price


Selling_price = input("Enter Selling Price : ")
mybolt = Bolt(conf.api_key, conf.device_id)
sms = Sms(conf.SSID, conf.AUTH_TOKEN, conf.TO_NUMBER, conf.FROM_NUMBER)
mailer = Email(conf.MAILGUN_API_KEY, conf.SANDBOX_URL, conf.SENDER_MAIL, conf.RECIPIENT_MAIL)

while True:
    c_price = get_bitcoin_price()
    print(get_bitcoin_price(), time.ctime())
    if c_price >= Selling_price:
        # Enable Buzzer
        response_buzzer = mybolt.digitalWrite('0', 'HIGH')
        print(response_buzzer)
        # Send SMS
        response_SMS = sms.send_sms("The Bitcoin selling price is now : " + str(c_price))
        # Send Mail
        response_mail = mailer.send_email("PRICE ALERT", "The Bitcoin selling price is now : " + str(c_price))
        # Send Telegram Alert
        message = 'Alert! Price is now : ' + str(c_price)
        telegram_status = send_telegram_message(message)
        print("This is the Telegram status:", telegram_status)
    else:
        response = mybolt.digitalWrite('0', 'LOW')
        print(response)
    time.sleep(297)
