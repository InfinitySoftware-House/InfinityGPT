import math
import sys
import torch
from torch import nn
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import Dataset
import os
from colorama import init as colorama_init
from colorama import Fore

colorama_init()

device = torch.device("cuda")
model_path = "C:/Users/gasto/Desktop/AI/InfinityGPT/InfinityLM/model/InfinityLM.pth"
model_exist = os.path.isfile(model_path)
saved_model = None
if model_exist:
    saved_model = torch.load(model_path)

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 512
        self.embedding_dim = 512
        self.num_layers = 4

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(device),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(device))
        
def get_sequence(dataset: Dataset, index: int):
    seq_lenght = args.sequence_length
    batch_size = args.batch_size
    
    sentence = dataset.sentences[index].split()
    sentence = [word for word in sentence]
    sentence = [dataset.word_to_index[word] for word in sentence]
    
    x = []
    y = []
    
    for i in range(batch_size):
        x.append(sentence[i:i+seq_lenght])
        y.append(sentence[i+1:i+seq_lenght+1])
    
    for i in range(batch_size):
        if len(x[i]) < seq_lenght:
              for _ in range(seq_lenght - len(x[i])): 
                x[i] = [0] + x[i]
        if len(y[i]) < seq_lenght:
            for _ in range(seq_lenght - len(y[i])): 
                y[i] = [0] + y[i]
        
    return (torch.tensor(x), torch.tensor(y))
   
def train(dataset, model: Model, optimizer, big_epoch, args):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    
    for batch, (x, y) in enumerate(dataloader):
        state_h, state_c = model.init_state(args.sequence_length)
        for epoch in range(1, args.max_epochs + 1):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x.to(device), (state_h, state_c))
            
            loss = criterion(y_pred.transpose(1, 2), y.to(device))

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
            #print every 100 epochs 
            if args.print_every > 0 and epoch % args.print_every == 0:
                print(f"epoch per batch: {epoch}, batch n°: {batch}, loss: {'{:.6f}'.format(loss.item())}")
            
        save_model(model, optimizer)
        print(Fore.YELLOW + "=========================================================" + Fore.WHITE)
        
def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    assert logits.dim() == 1
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value

    return logits

def apply_repeat_penalty(last_word_logits, generated_words, repeat_penalty):
    for word_idx in set(generated_words):
        last_word_logits[word_idx] /= repeat_penalty
    return last_word_logits

def predict(dataset, model, text, next_words=100, sentence_index=0):
    model.eval()
    sys.stdout.write(text)
    sys.stdout.write(' ')
    sys.stdout.flush()
    
    words = text.split(' ')
    words_index = []
    state_h, state_c = model.init_state(len(words))
    
    for i in range(0, next_words):
        words_index = [dataset.word_to_index[w] for w in words[i:]]
        x = torch.tensor([words_index])
        y_pred, (state_h, state_c) = model(x.to(device), (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        last_word_logits = apply_repeat_penalty(last_word_logits, words_index, repeat_penalty=1.5)
        last_word_logits = top_p_filtering(last_word_logits, top_p=0.9)
        
        p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        # if words_index[-1] == word_index:
        #     continue
        next_word = dataset.index_to_word[word_index]
        if next_word in dataset.stop_words:
            break
        else:
            words.append(next_word)
            if words_index[-1] != word_index:
                sys.stdout.write(next_word)
                sys.stdout.write(' ')
                sys.stdout.flush()
            
    # eval_percentage = '{:.2f}%'.format(EILM.eval(dataset.sentences, " ".join(words), sentence_index))
    # print("\n" + Fore.GREEN + "Eval: " + eval_percentage + Fore.YELLOW)
            
def save_model(model, optimizer):
    torch.save({
        "model_dict": model.state_dict(),
        "opt_dict": optimizer.state_dict(),
        "index_to_word": dataset.index_to_word,
        "word_to_index": dataset.word_to_index,
        "stop_words": dataset.stop_words
        }, model_path)
    # print(Fore.GREEN + "Model saved!" + Fore.WHITE)

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--sequence-length', type=int, default=8)
parser.add_argument('--print-every', type=int, default=10)
parser.add_argument('--max-big-epochs', type=int, default=10)
parser.add_argument('--dataset_lenght', type=int, default=2)

args = parser.parse_args()
# write_csv()
dataset = Dataset(args)

if saved_model is not None:
    dataset.index_to_word = saved_model["index_to_word"]
    dataset.word_to_index = saved_model["word_to_index"]
    dataset.stop_words = saved_model["stop_words"]

model = Model(dataset)
model.to(device)
if saved_model is not None:
    model.load_state_dict(saved_model["model_dict"])

optimizer = optim.Adam(model.parameters(), lr=0.001)

if saved_model is not None:
    optimizer.load_state_dict(saved_model["opt_dict"])

os.system("cls")

torch.cuda.empty_cache()

for i in range(1, args.max_big_epochs + 1):
    print(Fore.RED + f"Big Epoch {i}/{args.max_big_epochs}" + Fore.WHITE)
    train(dataset, model, optimizer, i, args)

predict(dataset, model, text="insieme da una", next_words=100, sentence_index=0)
print("\n===========================================================================")
sys.stdout.write(Fore.WHITE)