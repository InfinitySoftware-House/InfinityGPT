import shutil
import requests
import os
from pathlib import Path
from tqdm import tqdm
from colorama import init as colorama_init
from colorama import Fore
from clint.textui import progress

colorama_init()

MODEL_DOWNLOAD_URL = "https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/resolve/main/ggml-alpaca-7b-q4.bin"
MODEL_NAME = "ggml-alpaca-7b-q4.bin"
MODEL_PATH = "models/ggml-alpaca-7b-q4.bin"

MODEL_PATH_LOCAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), MODEL_PATH)

Path(MODEL_PATH_LOCAL).parent.mkdir(parents=True, exist_ok=True)

def download_file():
    #check file exist
    if os.path.isfile(MODEL_PATH_LOCAL):
        return True
    else:
        command = input(Fore.RED + "You have to download the model first!\nType y to continue: ")
        if command == "y":
            response = requests.get(MODEL_DOWNLOAD_URL, stream=True)
            with open(MODEL_PATH_LOCAL, 'wb') as f:
                total_length = int(response.headers.get('content-length'))
                with tqdm.wrapattr(response.raw, "read", total=total_length, desc="")as raw:
                    # save the output to a file
                    with open(MODEL_PATH_LOCAL, 'wb')as output:
                        shutil.copyfileobj(raw, output)
            return True
        else:
            return False