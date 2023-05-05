from colorama import Fore
from colorama import init as colorama_init

colorama_init()

HELP = """  
Available commands: 
-start (start the bot)
-exit (exit the bot)
-help (show help)

Template arguments: 
-search: Use this to search on internet. (What's the weather in New York? -search) 
-code: Use this to create a code snippet. (How to create a function? -code) 
-wikipedia: Use this to search on wikipedia. (When did Barcelona won the Champions League? -wikipedia)
-document: Use this to read a document. (<action> "<path to document>" -document), eg: (explain it "path/to/file" -document)
"""
            
WELCOME = Fore.YELLOW + f"This is a test project made using LLaMA AI model (with 7 Billion params)." + Fore.GREEN + "\n-start: to start the bot."+Fore.CYAN+"\n-help: to find commands.\n" + Fore.RED + "-exit: to exit the bot."

NOT_VALID_FORMAT_DOCUMENT = Fore.RED + f"Sorry, I can't read this document." + Fore.WHITE

AVAILABLE_MODELS = Fore.LIGHTCYAN_EX + """

Available models:
1: Alpaca 7B (Faster, worst results)
2: Vicuna 7B (Slower, better results)

""" + Fore.WHITE

MODEL_SELECTION = Fore.LIGHTCYAN_EX + """

Available models:
1: Alpaca 7B (Faster, worst results)
2: Vicuna 7B (Slower, better results)

Select a model you would like to use, if not exist, you will download it.

""" + Fore.WHITE