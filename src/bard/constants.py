import os

def _get_current_file_directory():
   return os.path.abspath(__file__)

def get_path_parts(path: str):
   return path.replace('\\', '/').split('/')

project_root = os.path.join(*get_path_parts(_get_current_file_directory())[:-3])
config_path = os.path.join(project_root, 'config.json')

pad_token = 0
start_token = 1
end_token = 2
