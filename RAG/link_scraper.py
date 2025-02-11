import requests
from bs4 import BeautifulSoup
import json

illnesses = {}
with open ('RAG/nhs_illnesses.json', 'r') as f:
    illnesses = json.load(f)

def get_relavent_text(url):
    info = {}
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    for row in soup.select('h2,p'):
        print(row.text)
        # label = None
        # if row.name == 'h2':
        #     label = row.text
        # elif row.name == 'p':
        #     info[label].append(row.text)
    return info

# for letter in illnesses:
#     for illness in illnesses[letter]:
#         url = illness[1]
#         get_relavent_text(url)

print(get_relavent_text(illnesses['A'][0][1]))