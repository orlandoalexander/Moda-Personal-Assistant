from google.cloud import bigquery, storage
from google.oauth2 import service_account
from io import BytesIO
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io

credentials = service_account.Credentials.from_service_account_file('authenticate-gcs.json')

PROJECT_ID = 'lewagonbootcamp-371116'
BUCKET_NAME = 'moda-asos-images'
URL_SECTION = 'https://moda-api-service-u3dpfzvyuq-ew.a.run.app/section'
URL_PREDICT = 'https://moda-api-service-u3dpfzvyuq-ew.a.run.app/predict'



class ASOS():
    def __init__(self) -> None:
        pass

    def run(self):
        self._get_images()

    def _get_images(self):
        storage_client = storage.Client.from_service_account_json('authenticate-gcs.json')
        blobs = storage_client.list_blobs(BUCKET_NAME)
        self._product_images = {}
        images = []
        previous_id = ''
        for blob in blobs:
            id = blob.name.split('/')[0]
            if id == previous_id:
                images.append(blob)
            else:
                if len(images)>3:
                    self._product_images[previous_id] = images
                    images = [blob]
            previous_id = id
        self._get_product_image()

    def _get_product_image(self):
        for self._id, images in self._product_images.items():
            self._product_section = None
            self._product_image = None
            for index, image in enumerate(images):
                img = image.download_as_bytes()
                file = {'file': img}
                section = requests.post(url=URL_SECTION, files=file).json()['section']
                if section != 'outfit':
                    self._product_section = section
                elif section == 'outfit':
                    self._product_image = img
                    self._image_url_index = index
                elif section == 'full body':
                    self._product_image = img
                    self._product_section = section
                    self._image_url_index = index
                if self._product_section != None and self._product_image != None:
                    break
                else:
                    pass
            if self._product_section != None and self._product_image != None:
                self._classify_outfit()

    def _classify_outfit(self):
        bq_client = bigquery.Client(credentials= credentials,project=PROJECT_ID)
        file = {'file': self._product_image}
        prediction = requests.post(url=URL_PREDICT, files=file).json()['results']
        if ('upper' in prediction) and ('lower' in prediction): # 'outfit'
            print('outfit')
            upper = prediction['upper']
            lower = prediction['lower']
            upper_category = upper['category']
            upper_color = upper['color']
            upper_fit = upper['fit']
            upper_design = upper['design']
            upper_sleeves = upper['sleeves']
            upper_neckline = upper['neckline']
            upper_fabric = upper['fabric']
            lower_category = lower['category']
            lower_color = lower['color']
            lower_fit = lower['fit']
            lower_design = lower['design']
            lower_fabric = lower['fabric']
            bq_client.query(f"""
            UPDATE moda_data.moda_asos_data
            SET upper_category = '{upper_category}', upper_design = '{upper_design}', upper_sleeves = '{upper_sleeves}', upper_neckline = '{upper_neckline}', upper_fabric = '{upper_fabric}', upper_fit = '{upper_fit}', lower_category = '{lower_category}', lower_design = '{lower_design}', lower_fit = '{lower_fit}', lower_fabric = '{lower_fabric}', upper_color = '{upper_color}', lower_color = '{lower_color}'
            WHERE id = {self._id}
            """)
        else: # 'full body'
            full_category = prediction['category']
            full_color = prediction['color']
            full_fit = prediction['fit']
            full_design = prediction['design']
            full_sleeves = prediction['sleeves']
            full_neckline = upper['neckline']
            full_length = upper['length']
            full_fabric = lower['fabric']
            bq_client.query(f"""
            UPDATE moda_data.moda_asos_data
            SET full_category = '{full_category}', full_design = '{full_design}', full_sleeves = '{full_sleeves}', full_fit = '{full_fit}', full_neckline = '{full_neckline}', full_length = '{full_length}', full_fabric = '{full_fabric}', full_color = '{full_color}'
            WHERE id = {self._id}
            """)
        print(self._id)

        # target image
        result = bq_client.query(f"""
        SELECT url_image
        FROM moda_data.moda_asos_data
        WHERE id = {self._id}
        """)
        for row in result:
            img_urls = row.url_image.replace('[','').replace(']','').replace("'","").split(', ')

        target_img_url = img_urls[self._image_url_index]

        f = bq_client.query(f"""
        UPDATE moda_data.moda_asos_data
        SET display_image = '{target_img_url}'
        WHERE id = {self._id}
        """)


asos = ASOS()
asos.run()
