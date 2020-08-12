import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'R&D':2, 'Administration':9, 'Marketing':6, 'State':'New York'})

print(r.json())