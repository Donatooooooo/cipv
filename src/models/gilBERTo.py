from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from datasets import Dataset
import torch, sys, os, random
import json, numpy as np

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
from features.visualization import plot_cm, plot_loss

class GilBERTo:
    def __init__(self, texts, labels, seed: int = 42):
        self.labels = [
            "Dominante e Schiavo emotivo",
            "Persona violenta e Succube",
            "Vittimista e Croccerossina",
            "Manipolatore e Dipendente emotiva",
            "Controllore e Isolata",
            "Sadico-Crudele e Masochista",
            "Narcisista e Succube",
            "Perfezionista Critico e Insicura Cronica",
            "Psicopatico e Adulatrice",
            "Geloso-Ossessivo e Sottomessa"
        ]

        self.model_name = "idb-ita/gilberto-uncased-from-camembert"
        self.cache_dir = "./models/downloaded"
        self.output_dir = "./models/finetuned"
        self.seed = seed

        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self._set_seed()

        print("- Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir = self.cache_dir,
        )

        special_tokens = {
            "additional_special_tokens": ["[HIST]", "[CURR]", "[NEXT]", "[A]", "[B]"]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 
                       'unk_token') and self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if os.path.isdir(self.output_dir) and os.path.exists(
                os.path.join(self.output_dir, "model.safetensors")):
            model_path = self.output_dir
        else:
            model_path = self.model_name

        print(f"- Loading model from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels = len(self.labels),
            id2label = self.id2label,
            label2id = self.label2id,
            cache_dir = self.cache_dir
        )

        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.texts = texts
        self.labels = labels 
        return
        
    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def preprocess_function(self, examples):
        encoded = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None
        )
        encoded["labels"] = examples["label"]
        return encoded

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }

    def get_dataloaders(self, test_size=0.2, val_size=0.1):
        texts, labels = self.texts, self.labels
        label_ids = [self.label2id[label] for label in labels]
    
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, label_ids, test_size=test_size, random_state=self.seed, stratify=label_ids
        )
    
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.seed, stratify=y_temp
        )
    
        train_dataset = Dataset.from_dict(
            {
                "text": X_train,
                "label": y_train
                }
            )
        val_dataset = Dataset.from_dict(
            {
                "text": X_val,
                "label": y_val
                }
            )
        test_dataset = Dataset.from_dict(
            {
                "text": X_test,
                "label": y_test
                }
            )
    
        return train_dataset, val_dataset, test_dataset

    def train(self,
              num_epochs = 10,
              patience = 5,
              batch_size = 32,
              learning_rate = 2e-5,
              warmup = 0.1,
              weight_decay = 0.01
              ):

        train_dataset, val_dataset, test_dataset = self.get_dataloaders()

        train_tokenized = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        val_tokenized = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        test_tokenized = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names
        )

        training_args = TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            learning_rate = learning_rate,
            warmup_ratio = warmup,
            weight_decay = weight_decay,
            logging_dir = f"{self.output_dir}/logs",
            logging_steps = 25,
            eval_strategy = "steps",
            eval_steps = 100,
            save_strategy = "steps",
            save_steps = 100,
            load_best_model_at_end = True,
            save_total_limit = 2,
            metric_for_best_model = "eval_f1_macro",
            greater_is_better = True,
            fp16 = torch.cuda.is_available(),
            seed = self.seed,
            disable_tqdm = False,
            dataloader_drop_last = True,
            remove_unused_columns = False,
            max_grad_norm = 1.0,
            optim = "adamw_torch",
            adam_beta1 = 0.9,
            adam_beta2 = 0.999,
            adam_epsilon = 1e-8,
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        self.trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer = self.tokenizer,
            data_collator = data_collator,
            compute_metrics = self.compute_metrics,
            callbacks = [
                EarlyStoppingCallback(early_stopping_patience = patience)
                ]
        )

        print("- GilBERTo fine-tuning started")
        self.trainer.train()

        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        test_results = self.trainer.evaluate(test_tokenized)
        print("- Model evaluated")

        predictions = self.trainer.predict(test_tokenized)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        cm = confusion_matrix(labels, preds)
        plot_cm(cm, self.labels)
        plot_loss(self.trainer)

        with open("gilberto_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=4, ensure_ascii=False)
        print(json.dumps(test_results, indent=4))
        return self.model, self.tokenizer