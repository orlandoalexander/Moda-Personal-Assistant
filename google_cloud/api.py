import requests
import pandas as pd
import numpy as np

class API():
    def __init__(self, API_KEY) -> None:
        self._url_categories = "https://asos2.p.rapidapi.com/categories/list"
        self._url_products = "https://asos2.p.rapidapi.com/products/v2/list"
        self._querystring_categories = {"country":"US","lang":"en-US"}
        self._headers = {"X-RapidAPI-Key": API_KEY, "X-RapidAPI-Host": "asos2.p.rapidapi.com"}

    def get_products(self):
        self._cats = self._get_categories()
        products = []
        for cat in self._cats:
            querystring = {f"store":"US","offset":"0","categoryId":{cat},"limit":"48","sort":"freshness","lang":"en-UK"}
            response = requests.request("GET", self._url, headers=self._headers, params=self._querystring).json()
            products.append(response)

    def _get_categories(self):
        response = requests.request("GET", self._url, headers=self._headers, params=self._querystring_categories).json()
        brands = np.array(response['brands'][0:4]) # get all clothing brands
        cats = np.array([cat['link']['categoryId'] for brand in brands for cat in brand['children']])
        return cats




    API_KEY = 'e6d0546079msh1698fc71e2b4d4ep1e1163jsnecaf48e249c5'
