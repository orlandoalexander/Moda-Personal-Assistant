import requests

# url = 'http://lewagonbootcamp-371116.uc.r.appspot.com/init'
# response = requests.get(url=url)
# print(response.json())


url = 'https://moda-api-service-u3dpfzvyuq-ew.a.run.app/section'
file = {'file': open('/Users/orlandoalexander/Desktop/images/outfit7.png', 'rb')}
response = requests.post(url=url, files=file)
print(response.json())

# url = 'http://0.0.0.0:8000/predict'
# file = {'file': open('/Users/orlandoalexander/Desktop/images/outfit9.png', 'rb')}
# response = requests.post(url=url, files=file)
# print(response.json())
