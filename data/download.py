import requests
import os
import tempfile
import zipfile

DATA_URL = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'

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
   
      

if __name__ == "__main__":
   data()
