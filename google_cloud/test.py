import requests

# url = 'http://lewagonbootcamp-371116.uc.r.appspot.com/init'
# response = requests.get(url=url)
# print(response.json())


url = 'https://moda-api-service-u3dpfzvyuq-ew.a.run.app/predict'
file = {'file': open('/Users/orlandoalexander/Desktop/images/romper8.png', 'rb')}
response = requests.post(url=url, files=file)
print(response.json())
