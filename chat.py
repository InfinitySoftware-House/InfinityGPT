import sys
import torch
from torch import nn
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import Dataset
import json
import csv

device = torch.device("mps")

def write_csv():
    json_data = {}
    with open("InfinityGPT/data/dev-v2.0.json", "r") as f:
        json_data = json.load(f)["data"]
    f = open('InfinityGPT/data/train_squad_4m.csv', 'w', encoding='UTF8')
    writer = csv.writer(f)  
    header = ["qas"]
    writer.writerow(header)
    for data in json_data:
        for paragraph in data["paragraphs"]:
            for qa in paragraph["qas"]:
                question = qa["question"]
                if len(qa["answers"]) > 0:
                    answer = qa["answers"][0]["text"]
                elif len(qa["plausible_answers"]) > 0:
                     answer = qa["plausible_answers"][0]["text"]
                else:
                    continue
                if "?" in question:
                    question = question.replace("?", "")
                row = [question + "? " + answer + " <end>"]
                writer.writerow(row)
    f.close()

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 1024
        self.embedding_dim = 1024
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
            dropout=0.3,
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

def train(dataset, model, optimizer, args):
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
                print(f"epoch per batch: {epoch}, batch n°: {batch}, loss: {'{:.4f}'.format(loss.item())}")
            
            
def predict(dataset, model, text, next_words=100):
    model.eval()
    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    sys.stdout.write(text)
    sys.stdout.write(' ')
    sys.stdout.flush()
    
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x.to(device), (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        next_word = dataset.index_to_word[word_index]
        if next_word == "<end>":
            break
        else:
            words.append(next_word)
            sys.stdout.write(next_word)
            sys.stdout.write(' ')
            sys.stdout.flush()
            
def save_model(model, optimizer):
    torch.save({
        "model_dict": model.state_dict(),
        "opt_dict": optimizer.state_dict(),
        "index_to_word": dataset.index_to_word,
        }, "InfinityGPT/models/InfinityLM.pth")

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--sequence-length', type=int, default=8)
parser.add_argument('--print-every', type=int, default=10)
args = parser.parse_args()
# write_csv()
dataset = Dataset(args)
model = Model(dataset)
model.to(device)
# model.load_state_dict(torch.load("InfinityGPT/models/InfinityLM.pth")["model_dict"])
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer.load_state_dict(torch.load("InfinityGPT/models/InfinityLM.pth")["opt_dict"])

train(dataset, model, optimizer, args)
save_model(model, optimizer)
predict(dataset, model, text="Who gave their", next_words=512)