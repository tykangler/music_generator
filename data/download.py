import requests
import os
import tempfile
import zipfile
from itertools import chain
import json

DATA_URL = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'
COMPOSER_EPOCHS = ['Baroque', 'Classical', 'Early Romantic', 'Romantic', 
                   'Late Romantic', '20th Century', 'Post-War', '21st Century']
COMPOSER_URLS = [
   f'https://api.openopus.org/composer/list/epoch/{epoch}.json'
   for epoch in COMPOSER_EPOCHS
]

def data(output_dir: str=os.path.dirname(os.path.abspath(__file__))):
   print("downloading zip...")
   response = requests.get(DATA_URL, allow_redirects=True)
   print("finished downloading...")
   print(f"extracting to {output_dir}/")
   with tempfile.TemporaryFile() as temp_zip:
      temp_zip.write(response.content)
      with zipfile.ZipFile(temp_zip) as unzip:
         unzip.extractall(f"{output_dir}/")
   print(f"done extracting")
   
def composer_metadata(output_dir: str=f"{os.path.dirname(os.path.abspath(__file__))}/metadata"):
   print("downloading composers...")
   response_generator = (requests.get(url).json()['composers'] for url in COMPOSER_URLS)
   response_json = list(chain.from_iterable(response_generator))
   print("finished downloading...")
   print("writing...")
   with open(f'{output_dir}/composers.json', 'w+') as composer_file:
      json.dump(response_json, composer_file)
   print("done writing")

if __name__ == "__main__":
   data()
   composer_metadata()
