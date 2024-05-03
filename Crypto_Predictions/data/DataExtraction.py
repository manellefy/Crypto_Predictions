import requests
import pandas as pd
from datetime import datetime, timedelta
import os

api_key = '9363b81796ff5c0b9a31083e4b02c5c492ff81601b4da1fe3b53b80528eff7f0'

cryptos = ['BTC','ETH','LTC','SOL','BNB','MKR']
base_url = "https://min-api.cryptocompare.com/data/v2/histohour"


start_date = datetime.now() - timedelta(days=365)
start_ts = int(start_date.timestamp())

now = datetime.now()
folder_name = now.strftime("data_%Y-%m-%d_%H-%M-%S")
folder_path = os.path.join("C:\\Users\\asus\\Desktop\\Crypto_Predictions\\data", folder_name)
os.makedirs(folder_path, exist_ok=True)

for coin in cryptos:
    print(f"Extraction des données pour {coin}...")

    to_ts = int(datetime.now().timestamp())

    params = {
        'fsym': coin,
        'tsym': 'USD',
        'limit': 2000,  
        'toTs': to_ts,
        'api_key': api_key
    }

    responses = []

    while True:
        response = requests.get(base_url, params=params).json()
        data = response['Data']['Data']
        responses.extend(data)
        
        
        params['toTs'] = data[0]['time'] - 3600
        
        
        if data[0]['time'] < start_ts or len(data) < params['limit']:
            break

    
    df = pd.DataFrame(responses)
    df = df[df['time'] >= start_ts]
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time': 'date', 'open': 'price', 'high': 'high', 'low': 'low', 'volumefrom': 'volumefrom', 'volumeto': 'volumeto'}, inplace=True)
    df = df[['date', 'price', 'high', 'low', 'volumefrom', 'volumeto']]
    

    
    df.sort_values(by='date', ascending=True, inplace=True)
    #transformer la colonne date en datetime
    df["date"] = pd.to_datetime(df["date"])
    
    filename = os.path.join(folder_path, f"{coin}_data.csv")
    df.to_csv(filename, index=False)
    print(f"Les données pour {coin} ont été sauvegardées dans {filename}")
    
print(f"Toutes les données ont été sauvegardées dans le dossier {folder_path}")
    
    