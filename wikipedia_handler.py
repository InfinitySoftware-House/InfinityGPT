from colorama import Fore
import wikipedia


def get_summary(subject):
    try:
        wikipedia_result = wikipedia.summary(subject, auto_suggest=True, sentences=4)
        return wikipedia_result
    except wikipedia.exceptions.DisambiguationError  as err:
        print(Fore.RED + f"Disambiguation! You have to choose one: \n{err.options}")
        return False