from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk, json, random, spacy


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
STOP_WORDS = set(stopwords.words('italian'))
nlp = spacy.load("it_core_news_sm")


def text_processing(text, lematization = False):
    # Tokenization and Lowecase
    words = word_tokenize(text.lower())
    
    # No simbols, stopword, |words|>2
    words = [w for w in words if w.isalpha() 
                        and w not in STOP_WORDS and len(w) > 2]
    
    if lematization:
        doc = nlp(" ".join(words))
        words = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS]
    
    text = " ".join(words)
    return text


def clean_conversation(all_data, lematization): 
    pairs = []
    for conversation in all_data:
        full_text = " ".join(str(turn["text"]) for turn in conversation["dialogue"])
        cleaned_text = text_processing(full_text, lematization)
        pairs.append((cleaned_text, conversation["person_couple"]))
    return pairs


def conversation_loader(lematization = False):
    with open("./data/processed/toxic_conversations.json", "r", encoding="utf-8") as f:
        toxic_data = json.load(f)
        
    pairs = clean_conversation(toxic_data, lematization)
    
    with open("./data/processed/safe_conversations.json", "r", encoding="utf-8") as f:
        safe_data = json.load(f)
        
    pairs += clean_conversation(safe_data, lematization)
    dataset = toxic_data + safe_data

    random.shuffle(pairs)
    texts, labels = [], []
    for item in pairs:
        texts.append(item[0])
        labels.append(item[1])
    
    print(f"- Loaded {len(texts)} conversations")
    return dataset, texts, labels