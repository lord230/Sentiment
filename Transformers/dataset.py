# Created By LORD 

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import config
import pandas as pd

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


# ____________________________________________________
#                      ABSA DATASET
# ____________________________________________________


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

class SentimentABSADataset(Dataset):
    def __init__(
        self,
        texts,
        sentiment_labels,
        tokenizer_name="xlm-roberta-base",
        max_length=128
    ):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def encode_sentiment(self, label):
        # -1 → 0 (Negative), 0 → 1 (Neutral), 1 → 2 (Positive)
        if label == -1:
            return 0
        elif label == 0:
            return 1
        else:
            return 2

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.encode_sentiment(self.sentiment_labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        sarcasm_label = torch.tensor(0, dtype=torch.long)
        aspect_labels = torch.full((self.max_length,), -100, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentiment_labels": torch.tensor(sentiment, dtype=torch.long), #-> Pos , Neg , Neu
            "sarcasm_labels": sarcasm_label, #->> Sarcasm
            "aspect_labels": aspect_labels #-->ABSA
        }

import pandas as pd

def load_data(path, max_samples=None):
    if max_samples is not None:
        df = pd.read_csv(path, nrows=max_samples)
    else:
        df = pd.read_csv(path)

    # enforce correct columns
    assert "summary" in df.columns
    assert "Sentiment" in df.columns

    texts = df["summary"].astype(str).tolist()
    labels = df["Sentiment"].astype(int).tolist()

    return texts, labels

def create_dataloaders(
    dataset_path,
    test_size=0.2,
    batch_size=8,
    tokenizer_name="xlm-roberta-base",
    max_length=128,
    random_state=42,
    max_samples=None
):

    texts, labels = load_data(dataset_path, max_samples=max_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    train_dataset = SentimentABSADataset(
        texts=X_train,
        sentiment_labels=y_train,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    test_dataset = SentimentABSADataset(
        texts=X_test,
        sentiment_labels=y_test,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


