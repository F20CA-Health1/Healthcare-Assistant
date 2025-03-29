import requests
from bs4 import BeautifulSoup
import json
import datetime

illnesses = {}
with open ('RAG/nhs_illnesses.json', 'r') as f:
    illnesses = json.load(f)

def get_content(ls, content=[]):
    for i in range(len(ls)):
        section = {}
        if ls[i].name == "h2":
            heading = ls[i].text
            ps = []
            for j in range(i+1, len(ls)):
                if ls[j].name == "p":
                    ps.append(ls[j].text)
                elif ls[j].name == "h2":
                    break
            if len(ps) == 0:
                continue
            section["heading"] = heading
            section["text"] = ps
            content.append(section)
            
            get_content(ls[j:], content)
            break
    return content


# goes to the given websit and return a dictionaty with the headings and paragraphs
# {headings: [heading1, heading2, ...], paragraphs: [paragraph1, paragraph2, ...]}
def get_relavent_text(url):
    info = {}
    info['url'] = url
    info['retrieval_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    info['content'] = get_content(soup.find_all(["h2", "p"]))
    info['embedding']  = ""
    return info

print(get_relavent_text(illnesses['A'][0][1]))

"""
Expected output:
{'url': 'https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/abdominal-aortic-aneurysm/',
 'retrieval_date': '2025-03-10',
 'content': [{'heading': 'Overview', 'text': ['An abdominal aortic aneurysm (AAA) is a swelling (aneurysm) of the aorta – the main blood vessel that leads away from the heart, down through the abdomen to the rest of the body.']},
             {'heading': 'Symptoms', 'text': ['In most cases, an AAA causes no noticeable symptoms, but if it bursts, it\'s a medical emergency that can be fatal.']},
             {'heading': 'When to get medical advice', 'text': ['You should see your GP if you have symptoms of an AAA.']},
             {'heading': 'Causes', 'text': ['It\'s not clear exactly what causes AAAs.']}
            ],
 'embedding': ''
}
"""

