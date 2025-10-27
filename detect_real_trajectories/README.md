Scripts to process both real and artificial market data, and then train machine learning models to classify 30-day trajectories as either real or artificial.

This is deemed to be a first-principles type problem in handling market data, through which we may establish sensible working assumptions for subsequent modelling. The central question is: can machine learning discern between real 30-day trajectories in crypto, vs statistically comparable random walks?

2 models are used here - an MLP model and an LSTM model, both implemented via PyTorch.

Data was collected across hundreds of cryptocurrencies, on a daily close level, over the past 365 days (where available). 30-day windows were then randomly sampled, while statistically comparable trajectories for each were also built via Monte Carlo.

The MLP performs strongly at this classification task (>80% overall accuracy), whereas the LSTM underperforms (only slightly better than a random guess). This may be due to a lack of hyperparameter optimisation or similar, or may indicate that the classification task is best accomplished by capturing superficial overall trajectory phenomena, rather than deeper analysis of sequential information.

Data provided by [CoinGecko](https://www.coingecko.com/).

Please cite this repository, if any scripts are used.
