import ttra_main_functions as ttra
import time
from datetime import datetime


print('Clearing cookies')
ttra.clear_cookies()

print('Finding games')
ttra.find_games()

filename = 'data/active_games.csv'

print('Finding odds')

import pandas as pd
from tqdm import tqdm

df = pd.read_csv(filename)
urls = df['url'].tolist()

for url in tqdm(urls):
    ttra.scrape_and_analyse_odds(url)

print('Done')

