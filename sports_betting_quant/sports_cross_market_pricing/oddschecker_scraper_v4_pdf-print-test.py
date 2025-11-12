from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time


def headless_screenshot(url: str, output_path: str):
    chrome_options = Options()
    # A commonly used user-agent
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/90.0.4430.85 Safari/537.36")
    chrome_options.add_argument("--headless")  # run in headless mode

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        
        time.sleep(2)

        # Capture screenshot (full browser window)
        driver.save_screenshot(output_path)
        print(f"Screenshot saved to: {output_path}")
    finally:
        driver.quit()


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


if __name__ == '__main__':
    
    url = 'https://www.oddschecker.com/baseball/mlb/toronto-blue-jays-at-new-york-mets/winner'
    
    url = 'https://www.oddschecker.com/baseball/mlb/oakland-athletics-at-colorado-rockies/winner'
    
    url = 'https://www.oddschecker.com/table-tennis/petr-vojtech-v-vaclav-hruska/winner'
    
    url = 'https://www.oddschecker.com/table-tennis/miloslav-dvorak-v-miroslav-cyrus/winner'
    
    url = 'https://www.oddschecker.com/tennis/atp-marrakech/tallon-griekspoor-v-luciano-darderi/winner'
    
    filename = 'data/'+url.split('/winner')[0].split('/')[-1]+'.csv'
    
    initialise = True
    
    for x in range(300):
        
        content = get_page_source(url)
        
        print('saving as PDF.')
        headless_screenshot(url, 'screenshots/test_screenshot.png')
        
        import sys
        sys.exit()
        
        timestamp = time.time()
        
        print(x)
        
        pre_proc = content.split('data-name="')
        outcome_order = []
        
        for chunk in pre_proc[1:]:
            outcome = chunk.split('"')[0]
            if outcome not in outcome_order:
                outcome_order.append(outcome)
        
        #print(outcome_order)
        
        processed = content.split('data-o="')
        
        odds_dict = {}
        
        for i in range(len(processed)):
            
            try:
            
                rate = processed[i+1].split('"')[0]
                
                provider_code = processed[i].split('data-bk="')[-1].split('"')[0]
                
                #print(provider_code, rate)
                
                if provider_code not in list(odds_dict.keys()):
                    odds_dict[provider_code] = {team:None for team in outcome_order}
                    odds_dict[provider_code][outcome_order[0]] = rate
                else:
                    odds_dict[provider_code][outcome_order[1]] = rate
            
            except IndexError:
                
                break
            
            
        #print(odds_dict)
        
        if initialise:
            with open(filename, 'w') as f:
                f.write('timestamp,num')
                for provider_code in list(odds_dict.keys()):
                    for outcome in list(odds_dict[provider_code].keys()):
                        f.write(f',{provider_code} - {outcome}')
            initialise = False
            
        with open(filename, 'a') as f:
            f.write(f'\n{timestamp},{x}')
            for provider_code in list(odds_dict.keys()):
                for outcome in list(odds_dict[provider_code].keys()):
                    f.write(f',{odds_dict[provider_code][outcome]}')
        
        time.sleep(20)


