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

model_ubuntu = "ggml-alpaca-7b-q4.bin"
model_mac = "/Users/francesco/Desktop/Progetti/AI/Models/ggml-alpaca-7b-q4.bin"
# model_mac = "/Users/francesco/Desktop/Progetti/AI/Models/gpt4-x-alpaca-13b-native-ggml-model-q4_0.bin"
# model_mac = "/Users/francesco/Desktop/Progetti/AI/Models/ggml-model-q4_0_13B.bin"
# model_mac = "/Users/francesco/Desktop/Progetti/AI/Models/ggml-vic13b-q5_1.bin"

prompt_thinking = PromptTemplate(template=Te.Templates.Thinking(), input_variables=["tools"])
prompt_google = PromptTemplate(template=Te.Templates.Search(), input_variables=["question"])
prompt_result = PromptTemplate(template=Te.Templates.Generate(), input_variables=["question", "result"])
prompt_wikipedia = PromptTemplate(template=Te.Templates.Wikipedia(), input_variables=["question"])

prompt_chat = PromptTemplate(template=Te.Templates.Chat(), input_variables=["message", "history", "name"])
prompt_code = PromptTemplate(template=Te.Templates.Code(), input_variables=["input"])
prompt_document = PromptTemplate(template=Te.Templates.Document(), input_variables=["input", "action"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager
model_ubuntu = os.path.join(os.path.dirname(__name__), model_ubuntu)

if mh.download_file():
    llm = LlamaCpp(model_path=mh.MODEL_PATH_LOCAL, callback_manager=callback_manager, verbose=True, n_ctx=2048, max_tokens=1024, streaming=True)
else:
    print("You need to download the model in order to use the bot.")
    quit()
    
clear_terminal()
chain = LLMChain(prompt=prompt_google, llm=llm)

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
        print(Fore.WHITE)
        response = chat_chain.run(result=search_results, question=user_input)
        return response
    if "-code" in user_input:
        user_input = user_input.replace("-code", "").strip()
        if u.check_valid_command(user_input):
            return False
        chat_chain.prompt = prompt_code
        print(Fore.GREEN)
        response = chat_chain.run(input=user_input)
        return response
    if "-wikipedia" in user_input:
        user_input = user_input.replace("-wikipedia", "").strip()
        if u.check_valid_command(user_input):
            return False
        subject = u.get_subject_phrase(nlp(user_input))
        chat_chain.prompt = prompt_result
        print(Fore.WHITE)
        wikipedia_result = w.get_summary(subject)
        response = chat_chain.run(result=wikipedia_result, question=user_input)
    if "-document" in user_input:
        user_input = user_input.replace("-document", "").strip()
        pattern = r'"(.*?)"'
        path = re.search(pattern, user_input)
        if path is not None:
            path = path.group(1)
            if u.check_valid_command(path, command="-document"):
                return False
            action = user_input.replace(path, "").strip()
            texts = d.get_document_text(path)
            if texts is None:
                print(c.NOT_VALID_FORMAT_DOCUMENT)
                return False
            document_summary = []
            chat_chain.prompt = prompt_document
            for text in texts:
                document_summary.append(chat_chain.run(input=text, action=action))
        
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
            bot_name = input(Fore.BLUE + "What's my name? " + Fore.WHITE)
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
                    print(Fore.RED)
                    chat_chain.prompt = prompt_chat
                    response = chat_chain.run(message=user_message, history=chat_history, name=bot_name)
                    chat_history.append({"role": "bot", "message": response})
                is_loading = False
                print("\n")            