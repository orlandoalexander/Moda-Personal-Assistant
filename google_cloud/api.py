import requests
import pandas as pd
import numpy as np
from google.cloud import storage

BUCKET_NAME = 'moda-data'
FILE_NAME = 'moda_asos_data.cvs'

class API():
    def __init__(self, API_KEY, BUCKET_NAME, FILE_NAME) -> None:
        self._url_categories = "https://asos2.p.rapidapi.com/categories/list"
        self._url_products = "https://asos2.p.rapidapi.com/products/v2/list"
        self._headers = {"X-RapidAPI-Key": API_KEY, "X-RapidAPI-Host": "asos2.p.rapidapi.com"}
        self._bucket_name = BUCKET_NAME
        self._file_name = FILE_NAME

    def get_products(self):
        # self._cats = self._get_categories()
        with open('exclude_items.txt', 'r') as f:
            self._exclude = f.readlines()
        print(self._exclude)

        # products = []
        # for cat in self._cats[:2]:
        #     querystring = {f"store":"US","offset":0,"categoryId":{cat},"limit":48,"sort":"freshness","lang":"en-UK"}
        #     response = requests.request("GET", self._url_products, headers=self._headers, params=querystring).json()
        #     product_count = response['itemCount']
        #     query_vals = [(48,48)]
        #     while sum(query_vals[-1]) < product_count:
        #         diff = (product_count-(query_vals[-1][0]+48))
        #         if diff < 48:
        #             n = diff%48
        #         query_vals.append((query_vals[-1][0]+48,n))
        #     query_vals.pop(0)
        #     for query_val in query_vals:
        #         querystring = {f"store":"US","offset":query_val[0],"categoryId":{cat},"limit":query_val[1],"sort":"freshness","lang":"en-UK"}
        #         response = requests.request("GET", self._url_products, headers=self._headers, params=querystring).json()
        #     product = [{'id':product['id'],'name':product['name'],'price_current':product['price']['current']['value'],'price_previous':product['price']['previous']['value'],'marked_down':product['price']['isMarkedDown'],'outlet':product['price']['isOutletPrice'],'selling_fast':product['isSellingFast'],'brand':product['brandName'], 'url':'https://www.asos.com/us/'+product['url'], 'url_image': [product['imageUrl']]+[product['imageUrl'][:product['imageUrl'].index(f"{product['id']}-")]+f"{product['id']}-{num}" for num in range(2,5)]} for product in response['products']]
        #     products.extend(product)
        # self._df = pd.DataFrame(products)
        # self._save_to_bq()

    def _get_categories(self):
        querystring = {"country":"US","lang":"en-US"}
        response = requests.request("GET", self._url_categories, headers=self._headers, params=querystring).json()
        brands = np.array(response['brands'][0:4]) # get all clothing brands
        cats = np.array([cat['link']['categoryId'] for brand in brands for cat in brand['children']])
        return cats

    def _save_to_bq(self):
        client = storage.Client(project='lewagonbootcamp-371116')
        bucket = client.bucket(self._bucket_name)
        blob = bucket.blob(self._file_name)
        blob.upload_from_string(self._df.to_csv(index = False),content_type = 'csv')

API_KEY=''
def main(data, context):
  api_asos = API(API_KEY, BUCKET_NAME, FILE_NAME)
  api_asos.get_products()

api_asos = API(API_KEY, BUCKET_NAME, FILE_NAME)
api_asos.get_products()
