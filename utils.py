from colorama import Fore
from colorama import init as colorama_init
import os

colorama_init()

def get_subject_phrase(doc):
    res = ""
    for token in doc:
        if ("PROPN" in token.pos_):
            res += token.text + " "
    return res.strip()

def check_valid_command(user_input, command = None):
    if user_input == "":
        print(Fore.RED + "Please enter valid command." + Fore.WHITE)
        return False
    if command is not None:
        if command == "-document":
            if not os.path.exists(user_input):
                print(Fore.RED + "Please enter valid path." + Fore.WHITE)
                return False