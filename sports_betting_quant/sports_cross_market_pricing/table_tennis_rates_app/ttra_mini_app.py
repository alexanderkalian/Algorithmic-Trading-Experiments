import ttra_main_functions as ttra
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import pandas as pd

def run_task(log_widget, task_name):
    def task():
        log_widget.insert(tk.END, f'\n{task_name}...')
        log_widget.see(tk.END)
        extra = ''
        if task_name == 'Finding in-play games':
            num_games = ttra.find_games()
            extra = f'{num_games} in-play games found.'
        elif task_name == 'Clearing cookies':
            ttra.clear_cookies()
        elif task_name == 'Scraping and analysing odds':
            filename = 'data/active_games.csv'
            df = pd.read_csv(filename)
            urls = df['url'].tolist()
            i = 0
            for url in urls:
                i += 1
                log_widget.insert(tk.END, f'\n{task_name}: game {i}/{len(urls)}')
                log_widget.see(tk.END)
                try:
                    min_win_1000_GBP, outcomes = ttra.scrape_and_analyse_odds(url)
                    min_win_1000_GBP = round(min_win_1000_GBP, 2)
                    game = f'{outcomes[0]} vs {outcomes[1]}'
                    if min_win_1000_GBP > 1000:
                        log_widget.insert(tk.END, f'\n    game {i}/{len(urls)} - arbitrage possible!\n    £1000.00 → £{min_win_1000_GBP}')
                        log_widget.insert(tk.END, f'\n    Game: {game}')
                        log_widget.see(tk.END)
                        # Finds out best providers.
                        best_providers = ttra.find_best_providers(url)
                        for outcome in outcomes:
                            key = f'{outcome} (best providers)'
                            providers = best_providers[key][0]
                            best_odds = best_providers[key][1]
                            log_widget.insert(tk.END, '\n-------------------------------------------')
                            log_widget.insert(tk.END, f'\n        Best odds ({outcome}): {round(best_odds,2)}')
                            prov_string = ', '.join(providers)
                            log_widget.insert(tk.END, f'\n        Providers:\n        {prov_string}')
                        log_widget.insert(tk.END, '\n-------------------------------------------')
                        log_widget.see(tk.END)
                    else:
                        log_widget.insert(tk.END, f'\n    game {i}/{len(urls)} - arbitrage *not* possible.\n    £1000.00 → £{min_win_1000_GBP}')
                        log_widget.see(tk.END)
                except TypeError:
                    log_widget.insert(tk.END, '\n    Error in output - moving on.')
                    log_widget.see(tk.END)
        log_widget.insert(tk.END, f'\n{task_name} - complete. {extra}\n')
        log_widget.see(tk.END)
    threading.Thread(target=task).start()  # run in background

def create_app():
    root = tk.Tk()
    root.title('Table Tennis Betting Arbitrage Opportunities - Dashboard')

    # Buttons
    btn_clear_cookies = tk.Button(root, text='Clear Cookies', command=lambda: run_task(log, 'Clearing cookies'))
    btn_find_games = tk.Button(root, text='Find In-Play Games', command=lambda: run_task(log, 'Finding in-play games'))
    btn_scrape_analyse = tk.Button(root, text='Scrape + Analyse Odds', command=lambda: run_task(log, 'Scraping and analysing odds'))
    btn_clear_cookies.pack(pady=5)
    btn_find_games.pack(pady=5)
    btn_scrape_analyse.pack(pady=5)

    # Console-like output area
    log = scrolledtext.ScrolledText(root, height=10, width=50)
    log.pack(padx=10, pady=10)

    root.mainloop()

if __name__ == '__main__':
    create_app()
