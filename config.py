
import json
import os
import sys

class config():
    def __init__(self):
        current_dir = os.getcwd()
        config_path = os.path.join(current_dir, 'config/property.json')

        with open(config_path) as json_file:
            self.json_data = json.load(json_file)
