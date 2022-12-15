import requests

# url = 'http://lewagonbootcamp-371116.uc.r.appspot.com/init'
# response = requests.get(url=url)
# print(response.json())


url = 'http://127.0.0.1:8000//predict'
file = {'file': open('/Users/orlandoalexander/Desktop/images/outfit.png', 'rb')}
response = requests.post(url=url, files=file)
print(response.json())
