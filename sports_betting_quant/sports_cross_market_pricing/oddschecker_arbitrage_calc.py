import pandas as pd
import numpy as np
from tqdm import tqdm


url = 'https://www.oddschecker.com/football/english/premier-league/fulham-v-liverpool/winner'

#url = 'https://www.oddschecker.com/football/english/premier-league/brentford-v-chelsea/winner'

url = 'https://www.oddschecker.com/football/english/premier-league/man-utd-v-man-city/winner'

url = 'https://www.oddschecker.com/football/germany/bundesliga/union-berlin-v-wolfsburg/winner'

filename = 'data/'+url.split('/winner')[0].split('/')[-1]+'.csv'


df = pd.read_csv(filename)
outcomes = df.keys().to_list()[2:]

print(outcomes)


print('Processing odds into decimal form:')

for o in tqdm(outcomes):
    
    odds_raw = df[o].to_list()
    
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

df['total_probability'] = df[[f'{o} (decimal)' for o in outcomes]].sum(axis=1)


print('Calculating minimum returns on £1,000:')

minimum_return_1000_GBP = []

for i in tqdm(range(len(df['total_probability'].to_list()))):
    
    total_prob = df['total_probability'].to_list()[i]
    
    allocations = (df.loc[i, [f'{o} (decimal)' for o in outcomes]]*1000 / total_prob).tolist()
    allocations = [round(float(a), 2) for a in allocations]
    
    min_return_1000 = None
    
    # Filters out cases where no odds could be obtained, or some other error.
    if not np.isnan(np.array(allocations)).any():
        returns = (allocations / df.loc[i, [f'{o} (decimal)' for o in outcomes]]).tolist()
        returns = [round(float(r), 2) for r in returns]
        min_return_1000 = min(returns)
    else:
        df.loc[i, 'total_probability'] = None
    
    minimum_return_1000_GBP.append(min_return_1000)

df['minimum_return_1000_GBP'] = minimum_return_1000_GBP


output_file = filename.split('.')[0]+'_returns_included.csv'

print('Saving to file.')

df.to_csv(output_file, index=False)

