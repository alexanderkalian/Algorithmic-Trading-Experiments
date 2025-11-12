import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date

# Get today's date in YYYY-MM-DD format
today = date.today().strftime('%Y-%m-%d')


#url = 'https://www.oddschecker.com/baseball/mlb/toronto-blue-jays-at-new-york-mets/winner'

url = 'https://www.oddschecker.com/baseball/mlb/oakland-athletics-at-colorado-rockies/winner'

url='https://www.oddschecker.com/table-tennis/zdenek-kasinski-v-pavel-berdych/winner'

url = 'https://www.oddschecker.com/table-tennis/josef-rossler-v-ales-krejci/winner'

url = 'https://www.oddschecker.com/table-tennis/jiri-louda-v-marcel-pikous/winner'

#filename = 'data/'+url.split('/winner')[0].split('/')[-1]+'.csv'
filename = 'data/'+url.split('/winner')[0].split('/')[-1]+'_'+today+'.csv'


df = pd.read_csv(filename)
odds_metrics = df.keys().to_list()[2:]

#print(odds_metrics)



odds_dict = {}


print('Processing odds into decimal form:')

for o in tqdm(odds_metrics):
    
    provider_code, outcome = o.split(' - ')
    
    if outcome not in list(odds_dict.keys()):
        odds_dict[outcome] = {}
    
    odds_raw = df[o].to_list()
    
    tot_len = len(odds_raw)
    
    odds_decimal = []
    
    for odds in odds_raw:
        try:
            if '/' in odds:
                odds_decimal.append(int(odds.split('/')[1])/(int(odds.split('/')[0])+int(odds.split('/')[1])))
            else:
                odds_decimal.append(int(odds))
        except TypeError:
            odds_decimal.append(None)
    
    df[f'{o} (decimal)'] = odds_decimal
    
    odds_dict[outcome][provider_code] = odds_decimal


print('Finding best rates and providers:')

for outcome in tqdm(odds_dict):
    
    best_odds = []
    best_providers = []
    
    for i in range(tot_len):
        
        comparison = []
        
        for provider_code in odds_dict[outcome]:
            
            comparison.append(odds_dict[outcome][provider_code][i])
        
        try:
            best_rate = min(val for val in comparison if val is not None)
            best_odds.append(best_rate)
            best_providers_temp = [prov for prov, rate in zip(odds_dict[outcome], comparison) if rate == best_rate]
            best_providers_string = '-'.join(best_providers_temp)
            best_providers.append(best_providers_string)
        except ValueError:
            best_odds.append(None)
            best_providers.append(None)
    
    df[f'{outcome} (best odds)'] = best_odds
    df[f'{outcome} (best providers)'] = best_providers


df['total_probability'] = df[[f'{o} (best odds)' for o in odds_dict]].sum(axis=1)


print('Calculating minimum returns on £1,000:')

minimum_return_1000_GBP = []

for i in tqdm(range(len(df['total_probability'].to_list()))):
    
    total_prob = df['total_probability'].to_list()[i]
    
    allocations = (df.loc[i, [f'{o} (best odds)' for o in odds_dict]]*1000 / total_prob).tolist()
    allocations = [round(float(a), 2) for a in allocations]
    
    min_return_1000 = None
    
    # Filters out cases where no odds could be obtained, or some other error.
    if not np.isnan(np.array(allocations)).any():
        returns = (allocations / df.loc[i, [f'{o} (best odds)' for o in odds_dict]]).tolist()
        returns = [round(float(r), 2) for r in returns]
        min_return_1000 = min(returns)
    else:
        df.loc[i, 'total_probability'] = None
    
    minimum_return_1000_GBP.append(min_return_1000)

df['minimum_return_1000_GBP'] = minimum_return_1000_GBP


output_file = filename.split('.')[0]+'_returns_included.csv'

print('Saving to file.')

df.to_csv(output_file, index=False)

