from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

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
    
    url = 'https://www.oddschecker.com/football/international-friendlies/usa-women-v-brazil-women/winner'
    
    url = 'https://www.oddschecker.com/football/english/premier-league/fulham-v-liverpool/winner'
    
    url = 'https://www.oddschecker.com/football/english/premier-league/brentford-v-chelsea/winner'
    
    url = 'https://www.oddschecker.com/football/english/premier-league/man-utd-v-man-city/winner'
    
    url = 'https://www.oddschecker.com/football/germany/bundesliga/union-berlin-v-wolfsburg/winner'
    
    filename = 'data/'+url.split('/winner')[0].split('/')[-1]+'.csv'
    
    initialise = True
    
    for x in range(300):
        
        content = get_page_source(url)
        
        # Print or process the content
        #print(content)
        
        timestamp = time.time()
        
        print(x)
        
        processed = content.split('Win Market')[3]
        
        processed1 = processed.split('</button>')[0]
        outcome1 = processed1.split('</p>')[0].split('>')[-1]
        odds1 = processed1.split('>')[-1]
        
        print(outcome1, odds1)
        
        processed2 = processed.split('</button>')[1]
        outcome2 = processed2.split('</p>')[0].split('>')[-1]
        odds2 = processed2.split('>')[-1]
        
        print(outcome2, odds2)
        
        processed3 = processed.split('</button>')[2]
        outcome3 = processed3.split('</p>')[0].split('>')[-1]
        odds3 = processed3.split('>')[-1]
        
        print(outcome3, odds3, '\n')
        
        if initialise:
            with open(filename, 'w') as f:
                f.write(f'timestamp,num,{outcome1},{outcome2},{outcome3}')
            initialise = False
            
        with open(filename, 'a') as f:
            f.write(f'\n{timestamp},{x},{odds1},{odds2},{odds3}')
        
        time.sleep(20)

