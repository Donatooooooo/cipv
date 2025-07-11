import pandas as pd
import json, re

def fix_broken_turns(dialogue):
    fixed = []
    i = 0
    while i < len(dialogue):
        current = dialogue[i]
        current_text = current["text"].strip()

        # Heuristic - short turn -> conversation is broken
        if (
            i + 1 < len(dialogue)
            and len(current_text) < 15
            and current["speaker"] != dialogue[i + 1]["speaker"]
        ):
            dialogue[i + 1]["text"] = current_text + " " + dialogue[i + 1]["text"]
            i += 1
        else:
            if fixed and current["speaker"] == fixed[-1]["speaker"]:
                fixed[-1]["text"] += " " + current_text
            else:
                fixed.append(current)
            i += 1
    return fixed

def parser(csv):
    by_names = 0
    by_quotes = 0
    by_spaces = 0

    conversations = []
    for _, row in csv.iterrows():
        person1 = row['name1']
        person2 = row['name2']
        couple = row['person_couple']
        explanation = row['explaination']
        raw_convo = row['conversation']

        ordered_turns = []
        if re.search(rf'\b({re.escape(person1)}|{re.escape(person2)})\b', raw_convo):
            # --- Name based parsing ---
            by_names += 1
            pattern = rf"({re.escape(person1)}|{re.escape(person2)})\s*:?[\s\n]*"
            segments = re.split(pattern, raw_convo)

            current_speaker = None
            for segment in segments:
                segment = segment.strip()
                if segment == person1 or segment == person2:
                    current_speaker = segment
                elif current_speaker and segment:
                    ordered_turns.append({
                        "speaker": current_speaker,
                        "text": segment
                    })
        else:
            # --- Quotes based parsing ---
            by_quotes += 1
            raw_convo_clean = re.sub(r'\s*\d+[\.\)\-]?\s*', '', raw_convo)
            quoted_phrases = re.findall(r'"([^"]+)"', raw_convo_clean)

            if quoted_phrases:
                for i, phrase in enumerate(quoted_phrases):
                    speaker = person1 if i % 2 == 0 else person2
                    ordered_turns.append({
                        "speaker": speaker,
                        "text": phrase.strip()
                    })
            else:
                # --- Spaces based parsing ---
                by_spaces += 1
                raw_convo_fallback = re.sub(r'["“”]', '', raw_convo_clean)
                fallback_turns = re.split(r'\s{2,}', raw_convo_fallback)
                fallback_turns = [t.strip() for t in fallback_turns if t.strip()]

                for i, turn in enumerate(fallback_turns):
                    speaker = person1 if i % 2 == 0 else person2
                    ordered_turns.append({
                        "speaker": speaker,
                        "text": turn
                    })

        turns = []
        for turn in ordered_turns:
            text = turn['text']
            text_clean = re.sub(r'[0-9]', '', text)
            text_clean = re.sub(r'[\\\/\"\']', '', text_clean)
            text_clean = re.sub(r'[’]', '', text_clean)
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()
            turns.append({
                "speaker": turn['speaker'],
                "text": text_clean
            })

        ordered_turns = fix_broken_turns(turns)
        conversation_entry = {
            "person_couple": couple,
            "name1": person1,
            "name2": person2,
            "dialogue": ordered_turns,
            "explanation": explanation
        }

        conversations.append(conversation_entry)

    # --- Statistics ---
    print("Name based parsing: ", by_names)
    print("Quotes based parsing: ", by_quotes)
    print("Spaces based parsing: ", by_spaces)

    return conversations

def get_dataset(input):
    csv = pd.read_csv(input)
    csv.columns = [col.strip() for col in csv.columns]
    return csv

def save_conversations(output, conversations):
    with open(output, "w", encoding="utf-8") as file:
        json.dump(conversations, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input = "./data/raw/explaination_toxic_conversation.csv"
    output = "./data/processed/toxic_conversations2.json"

    dataset = get_dataset(input)
    conversations = parser(dataset)
    save_conversations(output, conversations)