import requests
from bs4 import BeautifulSoup
import json

url = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

illnesses = soup.find_all('div', class_="az_list_indivisual")

dictionary = {}
for i in range(len(illnesses)):
    letter = illnesses[i]
    ls = []
    for j in range(len(letter.find_all('a'))):
        ls.append((letter.find_all('a')[j].text, letter.find_all('a')[j].get('href')))
    dictionary[letter.find_all('h2')[0].text] = ls

# write the dictionary to a json file
with open('nhs_illnesses.json', 'w') as f:
    json.dump(dictionary, f)
