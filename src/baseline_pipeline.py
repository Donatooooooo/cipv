from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from features.preprocessing import conversation_loader
from features.preprocessing import text_processing
from models.baseline import Baseline
from src.data.examples import get_examples

def closest_explanation(query, corpus):
    texts = [item["text"] for item in corpus]
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([query] + texts)

    cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    idx = np.argmax(cosine_similarities)
    score = cosine_similarities[idx]

    return corpus[idx]["explanation"], score

def baseline_pipeline():
    dataset, texts, labels = conversation_loader(lematization=True)
    baseline = Baseline()
    baseline.train(texts, labels)


    for tester in get_examples():
        print("\n", tester)
        text = text_processing(tester, lematization=True)
        pred_label, pred_proba = baseline.inference([text])
        print(f"Predicted label: {pred_label} - probability: {pred_proba}")
        
        matched = []
        for conversation in dataset:
            if conversation["person_couple"] == pred_label:
                matched.append(
                    {
                        "text": text_processing(" ".join(str(turn["text"]) 
                                    for turn in conversation["dialogue"]), lematization=True),
                        "explanation": conversation["explanation"]
                    }
                )

        explanation, score = closest_explanation(text, matched)
        print(f"======= Explanation -> score: {score} =======")
        print(explanation, "\n")

if __name__ == "__main__":
    baseline_pipeline()