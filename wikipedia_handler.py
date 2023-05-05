from colorama import Fore
import wikipedia


def get_summary(subject):
    try:
        wikipedia_result = wikipedia.page(title=subject, auto_suggest=False)
        return wikipedia_result.summary
    except wikipedia.exceptions.DisambiguationError  as err:
        print(Fore.RED + f"Disambiguation! You have to choose one: \n{err.options}")
        return False