import sys
import llama_cpp
from colorama import init as colorama_init
from colorama import Fore
import time
import templates as t

colorama_init()

# llm = llama_cpp.Llama(model_path="InfinityGPT/models/ggml-alpaca-7b-q4.bin", verbose=False)

messages = []

# messages.append(llama_cpp.ChatCompletionMessage(role="system", content="You are an helpful AI assistant."))

# while True:
#     message = input(Fore.BLUE + "User: ")
   
#     messages.append(llama_cpp.ChatCompletionMessage(role="user", content=message))

#     output = llm.create_chat_completion(messages)
#     sys.stdout.write(Fore.RED)
    
#     for char in output["choices"][0]["message"]["content"]:
#         sys.stdout.write(char)
#         sys.stdout.flush()
#         time.sleep(0.03)
        
#     print()
    
    
import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import templates as Te
from colorama import init as colorama_init
from colorama import Fore, Style
from duckduckgo_search import ddg
import spacy
import commands as c
import document as d
import re
import utils as u
import wikipedia_handler as w
import model_handler as mh
from bs4 import BeautifulSoup
import requests
import llama_cpp

colorama_init()

start = False
nlp = spacy.load('en_core_web_sm')

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

# prompt_thinking = PromptTemplate(template=Te.Templates.Thinking(), input_variables=["tools"])
# prompt_google = PromptTemplate(template=Te.Templates.Search(), input_variables=["question"])

# prompt_wikipedia = PromptTemplate(template=Te.Templates.Wikipedia(), input_variables=["question"])

# prompt_chat = PromptTemplate(template=Te.Templates.Chat(), input_variables=["message", "history", "name"])
# prompt_code = PromptTemplate(template=Te.Templates.Code(), input_variables=["input"])
# prompt_document = PromptTemplate(template=Te.Templates.Document(), input_variables=["input", "action"])

clear_terminal()

callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()])
bot_name = input(Fore.BLUE + "What's my name? " + Fore.WHITE)
mh.bot_name = bot_name

callback_manager = AsyncCallbackManager([Te.TokenHandler(bot_name=bot_name, time_end=0, time_start=0, total_time=0)])

MODEL_PATH = mh.download_file()
if MODEL_PATH is not None:
    llm = llama_cpp.Llama(model_path="InfinityGPT/models/ggml-alpaca-7b-q4.bin", verbose=False, n_ctx=4096)
    # llm = llama_cpp.Llama(model_path="InfinityGPT/models/ggml-vic7b-q5_0.bin")
    # llm = LlamaCpp(model_path="/Users/francesco/Desktop/Progetti/UniSym/InfinityGPT/models/ggml-gpt4all-j-v1.3-groovy.bin", callback_manager=callback_manager, verbose=True, n_ctx=8096, max_tokens=2048, streaming=True)
clear_terminal()
# chat_chain = LLMChain(prompt=prompt_chat, llm=llm)

chat_history = []

def start_commands(command):
    if "-start" in command:
        if u.check_valid_command(command):
            return False
        return True
    if "-exit" in command:
        if u.check_valid_command(command):
            return False
        print(Fore.RED + "Bye bye!")
        quit()
    if "-help" in command:
        if u.check_valid_command(command):
            return False
        print(Fore.BLUE + c.HELP + Fore.WHITE)
        return False
    return False

def find_commands(user_input):
    if "-search" in user_input:
        user_input = user_input.replace("-search", "").strip()
        if u.check_valid_command(user_input):
            return False
        search_results = search(question=user_input)
        prompt = Te.Templates.Generate().format(result=search_results, question=user_input)
        messages.append(llama_cpp.ChatCompletionMessage(role="user", content=prompt))
        response = llm.create_chat_completion(messages)
        return response
def get_url_text(url):
    request = requests.get(url)
    soup = BeautifulSoup(request.text, "html.parser")
    return soup.body.get_text()
    
def search(question):
    print(Fore.GREEN + f"Searching for results about '{question}'...\n")
        
    google_results = ddg(keywords=question, max_results=4)
    
    results_chain = []
    for result in google_results:
        results_chain.append(result["body"])
        
    # text = get_url_text(results_chain[0]).replace("\n", " ").strip()[:1000]

    print(Fore.GREEN + "Results found! Generating answer..." + Fore.BLUE)
        
    return "\n".join(results_chain)

if __name__ == "__main__":
    print(c.WELCOME)
    print(c.AVAILABLE_MODELS)
    while True:
        command = input("Command: ")
        response = start_commands(command)
        if response is True:
            start = response
        if start:
            messages.append(llama_cpp.ChatCompletionMessage(role="system", content=c.PRESENTATION.format(bot_name=bot_name)))
            response = llm.create_chat_completion(messages)
            mh.write_response(response)
            while True:
                message = input(Fore.BLUE + "User: ")
                messages.append(llama_cpp.ChatCompletionMessage(role="user", content=message))
                
                response = find_commands(message)
                
                sys.stdout.flush()
                if response is not None:
                     mh.write_response(response)
                else:
                    response = llm.create_chat_completion(messages)
                    mh.write_response(response)
                print("\n")            