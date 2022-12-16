from google.cloud import bigquery, storage
from google.oauth2 import service_account
from io import BytesIO
from PIL import Image
import requests

credentials = service_account.Credentials.from_service_account_file('authenticate-gcs.json')

PROJECT_ID = 'lewagonbootcamp-371116'
BUCKET_NAME = 'moda-asos-images'
URL = 'https://moda-api-service-u3dpfzvyuq-ew.a.run.app/predict'


class ASOS():
    def __init__(self) -> None:
        pass

    def run(self):
        self._get_images()
        self._get_product_image()

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
                    break
            previous_id = id

    def _get_product_image(self):
        for id, images in self._product_images.items():
            for image in images:
                img = BytesIO(image.download_as_bytes())
                file = {'file': img}
                section = requests.post(url=URL, files=file).json()['results']
                print(section)





asos = ASOS()
asos.run()
