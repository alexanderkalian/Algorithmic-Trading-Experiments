import pandas as pd


rolling_window = 60


filename = 'data/btc-usd-max.csv'

df = pd.read_csv(filename)

df['time'] = pd.to_datetime(df['snapped_at'])

df = df[['time','price']].sort_values('time')

df['perc_change'] = df['price'].pct_change()

df = df.dropna()

df['roll_avg'] = df['perc_change'].rolling(window=rolling_window).mean()
df['roll_std'] = df['perc_change'].rolling(window=rolling_window).std()

df = df.dropna()

print(df.head())

import os

output_directory = 'plots'
output_file = f'{output_directory}/btc_{rolling_window}-day_rolling_stats.png'
os.makedirs(output_directory, exist_ok=True)

halvings = ['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20']

import matplotlib.pyplot as plt

plt.plot(df['time'], df['roll_avg'], label=f'{rolling_window}-Day Rolling Avg')
plt.plot(df['time'], df['roll_std'], label=f'{rolling_window}-Day Rolling Std')

for d, lbl in zip(pd.to_datetime(halvings), ['2012', '2016', '2020', '2024']):
    if lbl == '2024':
        plt.axvline(d, linestyle=':', linewidth=1.2, alpha=0.8, color='gray', label='Halving Event')
    else:
        plt.axvline(d, linestyle=':', linewidth=1.2, alpha=0.8, color='gray')

plt.tight_layout()

plt.xlabel('Date')
plt.ylabel(f'{rolling_window}-Day Rolling Daily Price Action Stats\n(% Changes, in Fraction Form)')

plt.legend()
plt.savefig(output_file, dpi=500, bbox_inches='tight')
plt.show()

print(df.tail())



