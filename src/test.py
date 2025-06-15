import json
import glob
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ----- TF-IDF + SVM Baseline -----
def tfidf_svm_baseline(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    texts, labels = [], []
    for conversation in all_data:
        full_text = " ".join([turn["text"] for turn in conversation["dialogue"]])
        texts.append(full_text)
        labels.append(conversation["person_couple"])

    print(f"Caricate {len(texts)} conversazioni.")

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LinearSVC()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    print("\nTF-IDF + SVM Classification Report:\n")
    print(classification_report(y_test, y_pred))

# ----- DistilBERT Fine-tuning with Time-aware Input -----
class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.encodings = []
        self.labels = []
        for conv in data:
            dialogue = conv["dialogue"]
            full_text = " [SEP] ".join([f"{turn['speaker']}: {turn['text']}" for turn in dialogue])
            encoding = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
            self.encodings.append(encoding)
            self.labels.append(label2id[conv["person_couple"]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def distilbert_finetuning(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    labels = list(set(conv["person_couple"] for conv in all_data))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = DialogueDataset(train_data, tokenizer, label2id)
    test_dataset = DialogueDataset(test_data, tokenizer, label2id)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels), id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir="./results",
        do_eval = True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    print("\nFine-tuning completato.\n")

    # Valutazione
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    y_true = [example['labels'].item() for example in test_dataset]
    print("\nDistilBERT Classification Report:\n")
    print(classification_report(y_true, preds, target_names=labels))

# Esegui entrambi i confronti
json_path = "./data/processed/conversations.json"
tfidf_svm_baseline(json_path)
distilbert_finetuning(json_path)