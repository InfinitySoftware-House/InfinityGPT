import shutil
import requests
import os
from pathlib import Path
from tqdm import tqdm
from colorama import init as colorama_init
from colorama import Fore
import commands as c

colorama_init()

MODEL_DOWNLOAD_URL_ALPACA = "https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/resolve/main/ggml-alpaca-7b-q4.bin"

MODEL_PATH_ALPACA = "models/ggml-alpaca-7b-q4.bin"

MODEL_PATH_ALPACA_LOCAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), MODEL_PATH_ALPACA)

Path(MODEL_PATH_ALPACA_LOCAL).parent.mkdir(parents=True, exist_ok=True)

def check_model_exist(model):
    #check if file exist
    if model == "1":
        if os.path.exists(MODEL_PATH_ALPACA_LOCAL):
            return MODEL_PATH_ALPACA_LOCAL
   
    return None

def download_file():
    #check if file exist
    # model_selection = input(c.MODEL_SELECTION)
    model_chosed = check_model_exist("1")
    
    if model_chosed is not None:
        return model_chosed
    
    #keep 1, when there will be new models uncomment line 30
    match "1":
        case "1":
            print(Fore.GREEN + "Alpaca model will be downloaded")
            model_path = MODEL_PATH_ALPACA_LOCAL
            response = requests.get(MODEL_DOWNLOAD_URL_ALPACA, stream=True)
        case _:
            print(Fore.RED + "Wrong selection")
            quit()
    
    total_length = int(response.headers.get('content-length'))
    with tqdm.wrapattr(response.raw, "read", total=total_length, desc="")as raw:
        # save the output to a file
        with open(model_path, 'wb')as output:
            shutil.copyfileobj(raw, output)
    return True