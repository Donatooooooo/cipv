from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os, sys, json, numpy as np, joblib

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
from features.visualization import plot_cm

class Baseline:
    def __init__(self, config):
        self.best_params = None
        self.pipeline = None
        self.vectorizer = None
        self.model = None
        self.config = config
        
        if os.path.exists(f"./models/baseline_{config}.pkl"):
            print(f"- Loaded ./models/baseline_{config}")
            self.pipeline = joblib.load(f"./models/baseline_{config}.pkl")
            self.model = self.pipeline.named_steps["model"]
            self.vectorizer = self.pipeline.named_steps["tfidf"]
        return
    
    def train(self, texts, labels):
        if not self.pipeline:
            base_pipeline = Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("model", LogisticRegression(
                    max_iter=5000, 
                    solver="saga")
                )
            ])
            param_grid = {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                
                "model__C": [0.1, 1, 10, 30],
                "model__penalty": ["l2", "l1"]
            }

            outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
            all_best_params = []

            unique_labels = sorted(set(labels))
            cm_sum = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

            for train_idx, test_idx in outer_cv.split(texts, labels):
                X_tr = [texts[i] for i in train_idx]
                y_tr = [labels[i] for i in train_idx]
                X_te = [texts[i] for i in test_idx]
                y_te = [labels[i] for i in test_idx]

                gs = GridSearchCV(
                    base_pipeline,
                    param_grid,
                    cv=inner_cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1
                )
                gs.fit(X_tr, y_tr)

                y_pred = gs.predict(X_te)
                acc_scores.append(accuracy_score(y_te, y_pred))
                prec_scores.append(precision_score(y_te, y_pred, average="macro", zero_division=0))
                rec_scores.append(recall_score(y_te, y_pred, average="macro", zero_division=0))
                f1_scores.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
                
                all_best_params.append(gs.best_params_)

                cm = confusion_matrix(y_te, y_pred, labels=unique_labels)
                cm_sum += cm

            metrics = {
                "accuracy": f"{np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})",
                "precision_macro": f"{np.mean(prec_scores):.4f} (+/- {np.std(prec_scores):.4f})",
                "recall_macro": f"{np.mean(rec_scores):.4f} (+/- {np.std(rec_scores):.4f})",
                "f1_macro": f"{np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})"
            }
            print("- Model Evaluation:")
            print(json.dumps(metrics, indent=4))

            os.makedirs("./reports/baseline", exist_ok=True)
            with open(f'./reports/baseline/baseline_{self.config}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)

            best_fold_idx = np.argmax(f1_scores)
            self.best_params_ = all_best_params[best_fold_idx]
            
            print(f"- Best fold: {best_fold_idx + 1} with F1 score: {f1_scores[best_fold_idx]:.4f}")
            print("- Best params from best fold:", json.dumps(self.best_params_, indent=4))

            self.pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=self.best_params_["tfidf__max_features"],
                    ngram_range=self.best_params_["tfidf__ngram_range"]
                )),
                ("model", LogisticRegression(
                    max_iter=5000,
                    C=self.best_params_["model__C"],
                    penalty=self.best_params_["model__penalty"],
                    solver="saga"
                ))
            ])
            
            plot_cm(cm_sum, unique_labels, "Baseline_CM")

            self.pipeline.fit(texts, labels)
            joblib.dump(self.pipeline, f"./models/baseline_{self.config}.pkl")
            print("- Final model trained and saved")
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