import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import templates as Te
from colorama import init as colorama_init
from colorama import Fore
from duckduckgo_search import ddg
import spacy
import commands as c
import document as d
import re
import utils as u
import wikipedia_handler as w
import model_handler as mh

colorama_init()

start = False
nlp = spacy.load('en_core_web_sm')

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

prompt_thinking = PromptTemplate(template=Te.Templates.Thinking(), input_variables=["tools"])
prompt_google = PromptTemplate(template=Te.Templates.Search(), input_variables=["question"])
prompt_result = PromptTemplate(template=Te.Templates.Generate(), input_variables=["question", "result"])
prompt_wikipedia = PromptTemplate(template=Te.Templates.Wikipedia(), input_variables=["question"])

prompt_chat = PromptTemplate(template=Te.Templates.Chat(), input_variables=["message", "history", "name"])
prompt_code = PromptTemplate(template=Te.Templates.Code(), input_variables=["input"])
prompt_document = PromptTemplate(template=Te.Templates.Document(), input_variables=["input", "action"])

clear_terminal()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
bot_name = input(Fore.BLUE + "What's my name? " + Fore.WHITE)
callback_manager = CallbackManager([Te.TokenHandler(bot_name=bot_name, time_end=0, time_start=0, total_time=0)])

if mh.download_file():
    llm = LlamaCpp(model_path=mh.MODEL_PATH_LOCAL, callback_manager=callback_manager, verbose=True, n_ctx=2048*10, max_tokens=2048, streaming=True)
else:
    print("You need to download the model in order to use the bot.")
    quit()
    
clear_terminal()
chat_chain = LLMChain(prompt=prompt_chat, llm=llm)

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
        chat_chain.prompt = prompt_result
        search_results = search(question=user_input)
        response = chat_chain.run(result=search_results, question=user_input)
        return response
    if "-code" in user_input:
        user_input = user_input.replace("-code", "").strip()
        if u.check_valid_command(user_input):
            return False
        chat_chain.prompt = prompt_code
        response = chat_chain.run(input=user_input)
        return response
    if "-wikipedia" in user_input:
        user_input = user_input.replace("-wikipedia", "").strip()
        if u.check_valid_command(user_input):
            return False
        subject = u.get_subject_phrase(nlp(user_input))
        
        print(Fore.WHITE)
        wikipedia_result = w.get_summary(subject)
        if wikipedia_result is not False:
            splitted_text = d.split_text(wikipedia_result)
            print(Fore.GREEN + "Results found! Generating answer..." + Fore.WHITE)
            chat_chain.prompt = prompt_result
            response = chat_chain.run(result=splitted_text[0], question=user_input)
            return response
        else:
            return wikipedia_result
    if "-document" in user_input:
        user_input = user_input.replace("-document", "").strip()
        pattern = r'"(.*?)"'
        path = re.search(pattern, user_input)
        if path is not None:
            path = path.group(1)
            if u.check_valid_command(path, command="-document"):
                return False
            action = user_input.replace(f'"{path}"', "").strip()
            texts = d.get_document_text(path)
            if texts is None:
                print(c.NOT_VALID_FORMAT_DOCUMENT)
                return False
            document_summary = []
            chat_chain.prompt = prompt_document
            document_summary.append(chat_chain.run(input=texts, action=action))
        
def search(question):
    print(Fore.GREEN + f"Searching for results about '{question}'...\n")
        
    google_results = ddg(keywords=question, max_results=2)
    
    results_chain = []
    for result in google_results:
        results_chain.append(result["body"])

    print(Fore.GREEN + "Results found! Generating answer..." + Fore.BLUE)
        
    return results_chain

if __name__ == "__main__":
    print(c.WELCOME)
    while True:
        command = input("Command: ")
        response = start_commands(command)
        if response is True:
            start = response
        if start:
            while True:
                chat_history.append({"role": "system", "message": "You are an assistant called: " + bot_name})
                user_message = input(Fore.BLUE + "Task: " + Fore.WHITE)
                if user_message.strip() == "":
                    print(Fore.RED + "Please enter valid task.")
                    continue
                chat_history.append({"role": "user", "message": user_message})
                is_loading = True
                response = find_commands(user_message)
                if response is None:
                    print(Fore.WHITE)
                    chat_chain.prompt = prompt_chat
                    response = chat_chain.run(message=user_message, history=chat_history, name=bot_name)
                    chat_history.append({"role": "bot", "message": response})
                is_loading = False
                print("\n")            