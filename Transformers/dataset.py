# Created By LORD 

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

def build_vocab(sentences):
    counter = Counter()
    for s in sentences:
        counter.update(s.lower().split())

    stoi = {"<pad>": 0, "<unk>": 1}
    for word in counter:
        stoi[word] = len(stoi)
    return stoi


def build_label_map(labels):
    unique = sorted(set(labels))
    return {label: i for i, label in enumerate(unique)}


def encode(sentence, stoi, max_len):
    tokens = sentence.lower().split()
    ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    ids = ids[:max_len]
    ids += [stoi["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids)


class TextDataset(Dataset):
    def __init__(self, sentences, labels, stoi, label_map, max_len=20):
        self.sentences = sentences
        self.labels = labels
        self.stoi = stoi
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        x = encode(self.sentences[idx], self.stoi, self.max_len)
        y = torch.tensor(self.label_map[self.labels[idx]])
        return x, y


def create_loaders(csv_file, batch_size=32, max_len=32):
    sentences = []
    labels = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.append(row["Sentence"].strip())
            labels.append(row["Category"].strip())

    stoi = build_vocab(sentences)
    label_map = build_label_map(labels)


    train_s, test_s, train_l, test_l = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    train_ds = TextDataset(train_s, train_l, stoi, label_map, max_len=max_len)
    test_ds = TextDataset(test_s, test_l, stoi, label_map, max_len=max_len)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, stoi, label_map
