import requests

url = 'http://127.0.0.1:8000/init'
response = requests.get(url=url)


url = 'http://127.0.0.1:8000/predict'
file = {'file': open('/Users/orlandoalexander/Desktop/images/dress.png', 'rb')}
response = requests.post(url=url, files=file)
print(response.json())


# params={'models':['landmarks']}
# url = 'http://127.0.0.1:8000/update_models'
# response = requests.get(url=url,params=params)
# print(response.json())
