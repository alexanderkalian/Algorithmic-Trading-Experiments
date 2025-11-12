#table_tennis_url = 'https://www.oddschecker.com/table-tennis/'

#table_tennis_url = 'https://www.oddschecker.com/tennis'


### CLEARS COOKIES.

def clear_cookies(table_tennis_url = 'https://www.oddschecker.com/table-tennis/'):
    
    # Import necessary libraries.
    from selenium import webdriver
    from datetime import datetime, timedelta
    from selenium.webdriver.chrome.options import Options
    
    # Create a new Chrome Options instance and set headless mode
    chrome_options = Options()
    # A commonly used user-agent
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/90.0.4430.85 Safari/537.36")
    chrome_options.add_argument('--headless')  # run browser in headless mode
    
    # Applies options.
    driver = webdriver.Chrome(options=chrome_options)
    # Accesses website.
    driver.get(table_tennis_url)
    
    # Access CDP
    cookies = driver.execute_cdp_cmd('Network.getAllCookies', {})['cookies']
    
    # Filter cookies set in last 24 hours
    now = datetime.utcnow()
    cutoff = now - timedelta(days=1)
    
    # Iterates through recent cookies and deletes them.
    for cookie in cookies:
        creation_time = datetime.utcfromtimestamp(cookie['expires'] - cookie['maxAge']) if 'maxAge' in cookie else None
        if creation_time and creation_time > cutoff:
            driver.delete_cookie(cookie['name'])
    
    # Quits the selenium webdriver instance.
    driver.quit()




### FINDS GAMES.

def find_games(table_tennis_url = 'https://www.oddschecker.com/table-tennis/'):
    
    # Import necessary libraries.
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    
    # Create a new Chrome Options instance and set headless mode
    chrome_options = Options()
    # A commonly used user-agent
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/90.0.4430.85 Safari/537.36")
    chrome_options.add_argument('--headless')  # run browser in headless mode
    
    # Set up driver
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(5000, 3000)  # Helps avoid overlays
    
    # Accesses website.
    driver.get(table_tennis_url)
    wait = WebDriverWait(driver, 10) # waits a little
    
    # Wait until td elements load
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "td.time.all-odds-click")))
    
    # Store results
    results = []
    
    # Loop through all td elements by index to re-fetch after going back
    td_elements = driver.find_elements(By.CSS_SELECTOR, "td.time.all-odds-click")
    print(f"Found {len(td_elements)} clickable time cells.")
    
    # Iterate through all elements.
    for index in range(len(td_elements)):
        # Re-locate all tds fresh (important after going back)
        td_elements = driver.find_elements(By.CSS_SELECTOR, "td.time.all-odds-click")
        td = td_elements[index]
    
        # Extract the nested span under a div
        try:
            nested_div = td.find_element(By.TAG_NAME, "div")
            time_span = nested_div.find_element(By.TAG_NAME, "span")
            match_time = time_span.text.strip()
        except Exception as e:
            match_time = "Unknown"
            print(f"Could not extract time for element {index}: {e}")
    
        # Scroll to view and click
        driver.execute_script("arguments[0].scrollIntoView();", td)
        #td.click()
        driver.execute_script("arguments[0].scrollIntoView(true);", td)
        time.sleep(0.2)  # Let it scroll
        driver.execute_script("arguments[0].click();", td)
    
        # Wait for redirect
        wait.until(lambda d: d.current_url != table_tennis_url)
        match_url = driver.current_url
        
        # Ignores this instance, if the game seems expired.
        html = driver.page_source
        if 'No Prices Yet' in html:
            driver.back()
            time.sleep(2)
            continue
        
        # Does this if only in-play games are sought out.
        if 'IN PLAY' not in match_time:
            break
        
        print(f"[{match_time}] → {match_url}")
        results.append((match_time, match_url))
    
        # Go back to main page and wait
        driver.back()
        time.sleep(2)
    
    # Initialises output file.
    output_file = 'data/active_games.csv'
    with open(output_file, 'w') as f:
        f.write('url')
    
    # Final printout
    print("\nFinal results:")
    for time_val, url in results:
        print(f"{time_val} → {url}")
        with open(output_file, 'a') as f:
            f.write(f'\n{url}')
        
    # Quits selenium webdriver instance.
    driver.quit()
    
    return len(results)



### SCRAPES AND ANALYSES ODDS.

def scrape_and_analyse_odds(url):
    
    # Imports necessary libraries.
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import time
    from datetime import date
    import os

    # Get today's date in YYYY-MM-DD format
    today = date.today().strftime('%Y-%m-%d')

    # Sub-function to get page source html.
    def get_page_source(url: str) -> str:
        '''
        Launches a headless browser, opens the given URL, and returns the page source.
        '''
        # Create a new Chrome Options instance and set headless mode
        chrome_options = Options()
        # A commonly used user-agent
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/90.0.4430.85 Safari/537.36")
        chrome_options.add_argument('--headless')  # run browser in headless mode

        # Create a new WebDriver instance using the Chrome driver
        driver = webdriver.Chrome(options=chrome_options)

        try:
            # Navigate to the given URL
            driver.get(url)
            
            time.sleep(2)

            # Extract the page source (HTML) for subsequent parsing/scraping
            page_source = driver.page_source
            
            #table = driver.find_element(By.ID, "myTable")
            
        finally:
            # Close the driver after we have retrieved the content
            driver.quit()

        return page_source

    # Carry out function.
    
    # Define filename.
    filename = 'data/'+today+'_'+url.split('/winner')[0].split('/')[-1]+'.csv'
    
    # Call function to get source html.
    content = get_page_source(url)
    
    # Obtain current timstamp.
    timestamp = time.time()
    
    # Split up source html in a way that is useful for mining possible outcomes.
    pre_proc = content.split('data-name="')
    # List to hold possible outcomes.
    outcome_order = []
    
    # Iterate through split apart source html chunks.
    for chunk in pre_proc[1:]:
        # Isolate a given outcome.
        outcome = chunk.split('"')[0]
        # If outcome is not already found, note it.
        if outcome not in outcome_order:
            outcome_order.append(outcome)
    
    #print(outcome_order)
    
    # Split up source html in a way that is useful for mining provider-specific odds.
    processed = content.split('data-o="')
    
    # Create a dictionary to hold provider-specific odds information.
    odds_dict = {}
    
    # Iterate through the split up chunks of the source html.
    for i in range(len(processed)):
        
        try:
            
            # Isolate odds rate information.
            rate = processed[i+1].split('"')[0]
            
            # Isolate code specific to provider.
            provider_code = processed[i].split('data-bk="')[-1].split('"')[0]
            
            #print(provider_code, rate)
            
            # If provider is not already known, adds to dictionary.
            if provider_code not in list(odds_dict.keys()):
                # Creates None value placeholder for different outcome odds.
                odds_dict[provider_code] = {team:None for team in outcome_order}
                # Updates this with first outcome's odds.
                odds_dict[provider_code][outcome_order[0]] = rate
            else:
                # Updates this with second outcome's odds.
                odds_dict[provider_code][outcome_order[1]] = rate
        
        except IndexError:
            
            break
        
    #print(odds_dict)
    
    # Creates output file, if it doesn't already exist.
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            # Adds csv file column name for timestamp.
            f.write('timestamp')
            # Adds column names for odds for all found providers and outcomes.
            for provider_code in list(odds_dict.keys()):
                for outcome in list(odds_dict[provider_code].keys()):
                    f.write(f',{provider_code} - {outcome}')
            for provider_code in list(odds_dict.keys()):
                for outcome in list(odds_dict[provider_code].keys()):
                    f.write(f',{provider_code} - {outcome} (decimal)')
            for outcome in outcome_order:
                f.write(f',{outcome} (best odds),{outcome} (best providers)')
            f.write(',total_probability,minimum_return_1000_GBP')
    
    # Writes data to output file.
    with open(filename, 'a') as f:
        # Update with timestamp information.
        f.write(f'\n{timestamp}')
        for provider_code in list(odds_dict.keys()):
            for outcome in list(odds_dict[provider_code].keys()):
                # Update with provider and outcome specific odds info.
                f.write(f',{odds_dict[provider_code][outcome]}')
    
    ## Calculates best rates.
    
    if len(outcome_order) != 2:
        print('Erronious number of outcomes found, for 2-player table tennis match')
        return False
    
    # Extracts lists of outcome-specific odds and providers.
    odds_outcome0 = [odds_dict[prov][outcome_order[0]] for prov in odds_dict.keys()]
    odds_outcome1 = [odds_dict[prov][outcome_order[1]] for prov in odds_dict.keys()]
    providers = list(odds_dict.keys())
    
    # Calculates decimal odds.
    
    # List to holds decimal odds.
    odds_outcome0_dec = []
    
    # Iterates through all odds in a given list.
    for odds in odds_outcome0:
        try:
            # Converts to decimal probability and appends to list.
            if '/' in odds:
                odds_outcome0_dec.append(int(odds.split('/')[1])/(int(odds.split('/')[0])+int(odds.split('/')[1])))
            else:
                odds_outcome0_dec.append(1/(int(odds)+1))
        # If there is an issue, odds are None.
        except ValueError:
            odds_outcome0_dec.append(None)
    
    # List to holds decimal odds.
    odds_outcome1_dec = []
    
    # Iterates through all odds in a given list.
    for odds in odds_outcome1:
        try:
            # Converts to decimal probability and appends to list.
            if '/' in odds:
                odds_outcome1_dec.append(int(odds.split('/')[1])/(int(odds.split('/')[0])+int(odds.split('/')[1])))
            else:
                odds_outcome1_dec.append(1/(int(odds)+1))
        # If there is an issue, odds are None.
        except ValueError:
            odds_outcome1_dec.append(None)
    
    # Writes decimal odds to file.
    with open(filename, 'a') as f:
        for odds_dec0, odds_dec1 in zip(odds_outcome0_dec, odds_outcome1_dec):
            f.write(f',{odds_dec0},{odds_dec1}')
    
    # Finds best odds for each outcome.
    best_odds_outcome0 = min(x for x in odds_outcome0_dec if x is not None)
    best_odds_outcome1 = min(x for x in odds_outcome1_dec if x is not None)
    
    # Finds associated best providers.
    
    # For outcome 0.
    best_providers_outcome0 = []
    for prov, odds in zip(providers, odds_outcome0_dec):
        if odds == best_odds_outcome0:
            best_providers_outcome0.append(prov)
    
    # For outcome 1.
    best_providers_outcome1 = []
    for prov, odds in zip(providers, odds_outcome1_dec):
        if odds == best_odds_outcome1:
            best_providers_outcome1.append(prov)
    
    # Finds summary strings.
    best_providers_string0 = '-'.join(best_providers_outcome0)
    best_providers_string1 = '-'.join(best_providers_outcome1)
    
    # Records best odds and providers to file.
    with open(filename, 'a') as f:
        f.write(f',{best_odds_outcome0},{best_providers_string0},{best_odds_outcome1},{best_providers_string1}')
    
    # Calculates total probability (via best odds.)
    tot_prob = best_odds_outcome0+best_odds_outcome1
    # Records this to file.
    with open(filename, 'a') as f:
        f.write(f',{tot_prob}')
    
    # Calculates minimum return on £1,000 optimally allocated between both sides.
    
    # Finds optimal allocations first (proportional to probability).
    optimal_allocations = [round(best_odds_outcome0*1000/tot_prob, 2), round(best_odds_outcome1*1000/tot_prob, 2)]
    
    # Calculates respective returns (divide by probability).
    expected_returns = [optimal_allocations[0]/best_odds_outcome0, optimal_allocations[1]/best_odds_outcome1]
    
    # Finds minimum of these two.
    minimum_return_1000_GBP = min(expected_returns)
    
    # Records this to file.
    with open(filename, 'a') as f:
        f.write(f',{minimum_return_1000_GBP}')
    
    return minimum_return_1000_GBP, outcome_order


### DICTIONARY OF BETTING PROVIDERS.

def betting_providers_dict():
    
    betting_providers = {'B3': 'bet365', 'SK': 'SkyBet', 'PP': 'PaddyPower', 
                         'WH': 'William Hill', 'EE': '888sport', 'FB': 'betfair', 
                         'VC': 'BetVictor', 'LD': 'Ladbrokes', 'UN': 'Unibet', 
                         'SX': 'Spreadex', 'FR': 'Betfred', 'KN': 'BetMGM', 
                         'BY': 'BoyleSports', 'OE': '10bet', 'S6': 'Star Sports', 
                         'PUP': 'PricedUP', 'SI': 'Sporting Index', 'LS': 'LiveScore Bet', 
                         'QN': 'QuinnBet', 'WA': 'Betway', 'CE': 'Coral', 
                         'N4': 'Midnite', 'G5': 'BetGoodwin', 'VT': 'VBET', 
                         'AKB': 'AK bets', 'BF': 'BetFair'}
    
    return betting_providers


### Finds best providers.

def find_best_providers(url):
    
    from datetime import date
    import pandas as pd
    
    # Get today's date in YYYY-MM-DD format
    today = date.today().strftime('%Y-%m-%d')
    
    # Define filename.
    filename = 'data/'+today+'_'+url.split('/winner')[0].split('/')[-1]+'.csv'
    
    # Define provider code dict.
    providers_dict = betting_providers_dict()
    
    # Open the data file and find the most recent best providers data.
    df = pd.read_csv(filename)
    headers = df.keys().tolist()
    # Builds dict of best providers.
    best_providers = {h:df[h].tolist()[-1] for h in headers if '(best providers)' in h}
    # Finds actual list of best providers.
    for outcome in best_providers:
        best_odds = df[outcome.replace('(best providers)', '(best odds)')].tolist()[-1]
        prov_codes = best_providers[outcome].split('-')
        best_providers[outcome] = [[providers_dict[p] for p in prov_codes], best_odds]
    
    return best_providers
    
    
    


### FINDS EXPIRED GAMES.
'''
def find_expired_games(url)
'''
