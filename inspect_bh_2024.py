import requests
from bs4 import BeautifulSoup
import re

url = 'https://www.bollywoodhungama.com/box-office-collections/filterbycountry/IND/2024'
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')

links = soup.find_all('a', href=True)
print(f"Found {len(links)} links:")
for a in links:
    text = a.text.strip()
    href = a['href']
    if '/movie/' in href and '/box-office/' in href:
        print(f"  {text} | {href}")
