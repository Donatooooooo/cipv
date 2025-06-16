from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os, json, numpy as np, joblib
import matplotlib.pyplot as plt
import seaborn as sns


class Baseline:
    def __init__(self):
        self.best_params = None
        self.pipeline = None
        self.vectorizer = None
        self.model = None
        
        if os.path.exists("./models/baseline.pkl"):
            print("- Loaded ./models")
            self.pipeline = joblib.load("./models/baseline.pkl")
            self.model = self.pipeline.named_steps["model"]
            self.vectorizer = self.pipeline.named_steps["tfidf"]
        return

    def train(self, texts, labels):
        if not self.pipeline:
            pipeline = Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("model", LogisticRegression(max_iter=5000)),
                ]
            )

            param_grid = {
                "tfidf__max_features": [5000, 10000],
                "model__C": [0.1, 1, 10, 30],
                "model__penalty": ["l2", "l1"],
                "model__solver": ["saga"],
            }

            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=10,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )

            grid_search.fit(texts, labels)
            self.best_params = grid_search.best_params_
            print("- Saved best hyperparameters")
            print(json.dumps(self.best_params, indent=4))
            with open('./reports/baseline/baseline_hyperparams.json', 'w') as f:
                json.dump(self.best_params, f, indent=4)

            self.pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=self.best_params[
                                "tfidf__max_features"
                            ],
                        ),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            C=self.best_params["model__C"],
                            penalty=self.best_params["model__penalty"],
                            solver=self.best_params["model__solver"],
                        ),
                    ),
                ]
            )

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        scores = cross_validate(
            self.pipeline, texts, labels, cv=kfold, scoring=scoring, n_jobs=-1
        )

        metrics = {
            "accuracy": 
                f"{scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std():.4f})",
            "precision_macro": 
                f"{scores['test_precision_macro'].mean():.4f} (+/- {scores['test_precision_macro'].std():.4f})",
            "recall_macro": 
                f"{scores['test_recall_macro'].mean():.4f} (+/- {scores['test_recall_macro'].std():.4f})",
            "f1_macro": 
                f"{scores['test_f1_macro'].mean():.4f} (+/- {scores['test_f1_macro'].std():.4f})"
        }

        print("- Model evaluated")
        print(json.dumps(metrics, indent=4))
        with open('./reports/baseline/baseline_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        y_pred = cross_val_predict(self.pipeline, texts, labels, cv=kfold, n_jobs=-1)
        cm = confusion_matrix(labels, y_pred, labels=sorted(set(labels)))
        plot(cm, sorted(set(labels)))

        self.pipeline.fit(texts, labels)
        self.model = self.pipeline.named_steps["model"]
        self.vectorizer = self.pipeline.named_steps["tfidf"]

        joblib.dump(self.pipeline, "./models/baseline.pkl")
        return
    
    def inference(self, texts):
        if not self.pipeline:
            print("! Model is not trained")
            return
        else:
            label = self.pipeline.predict(texts)[0]
            proba = self.pipeline.predict_proba(texts)[0]
            proba = max(proba)
        return label, proba


def plot(cm, class_labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.xticks(ticks=np.arange(len(class_labels)) + 0.5,
               labels=class_labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(class_labels)) + 0.5,
               labels=class_labels, rotation=0, va='center')

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig('./reports/figures/Baseline_CM.png')
    plt.show()