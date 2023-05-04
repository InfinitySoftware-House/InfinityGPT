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
            
WELCOME = Fore.YELLOW + f"This is a test project made using LLaMA AI model (with 7 Billion params), we are using DuckDuckGo to browse the internet." + Fore.GREEN + "\n-start: to start the bot."+Fore.CYAN+"\n-help: to find commands.\n" + Fore.RED + "-exit: to exit the bot.\n" + Fore.WHITE

NOT_VALID_FORMAT_DOCUMENT = Fore.RED + f"Sorry, I can't read this document." + Fore.WHITE