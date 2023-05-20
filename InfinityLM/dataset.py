import csv
import re
import torch
import pandas as pd
from collections import Counter
import wikipedia
import re

wikipedia.set_lang("it")

def get_wikipedia():
    page = wikipedia.page("Catalogo NGC completo - 1-999")
    for link in page.links[:50]:
        text = wikipedia.summary(link)      
        with open('C:/Users/gasto/Desktop/AI/InfinityGPT/data/train_2.txt', 'a', encoding="utf-8") as f:
            text = text.replace('\n', '')
            text += ' <endoftext> \n'
            f.write(text)
            print(f"Pagina salvata: {link}")
    print("Fine")

def clear_text(text):
    rx = re.compile('\W+')
    res = rx.sub(' ', text).strip()
    return res
    
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.stop_words = ["endoftext"]
        self.end_char = {}
        self.sentences = self.get_sentences()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        
        self.index_to_word[len(self.uniq_words)+1] = " "
        self.word_to_index[" "] = len(self.uniq_words)+1
        
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        with open('C:/Users/gasto/Desktop/AI/InfinityGPT/data/train_2.txt', newline='', encoding="utf-8") as f:
            txt = f.readlines()[:self.args.dataset_lenght]
            txt = " ".join(txt)
            txt = clear_text(txt)
            return txt.split(' ')
        
    def get_sentences(self):
        with open('C:/Users/gasto/Desktop/AI/InfinityGPT/data/train_2.txt', newline='', encoding="utf-8") as f:
            txt = f.readlines()
            txt = [x.replace('\r\n', '') for x in txt]
            return txt

    def get_uniq_words(self):
        with open('C:/Users/gasto/Desktop/AI/InfinityGPT/data/train_2.txt', newline='', encoding="utf-8") as f:
            txt = f.readlines()
            txt = " ".join(txt)
            txt = clear_text(txt)
            words = txt.split(' ')
            word_counts = Counter(words)
            return sorted(word_counts, key=word_counts.get, reverse=True)
        

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )