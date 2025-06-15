from jinja2 import Environment, FileSystemLoader
import google.generativeai as gemini
import time, json, os, tqdm, random

NUM_CONVERSATIONS = 800
API_CALL_DELAY = 0.3
MAX_RETRIES = 3

def generate_prompt(data):
    environment = Environment(
        loader = FileSystemLoader("./data/util")
        )
    template = environment.get_template("prompt_template.jinja")
    return template.render(data)

def generate_conversation(model, prompt, max_retries=5, delay=1.0):
    generation_config = {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "max_output_tokens": 2048,
        "response_mime_type": "text/plain",
    }

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            raw = response.text
            if not raw:
                print(f"[Attempt {attempt + 1}] Empty response. Retrying...")
                continue

            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw.removeprefix("```json").removesuffix("```").strip()

            try:
                parsed_response = json.loads(raw)
                return parsed_response
            except json.JSONDecodeError as e:
                print(f"[Attempt {attempt + 1}] JSON parsing error: {e}. Retrying...")

        except Exception as e:
            print(f"[Attempt {attempt + 1}] Error during generation: {e}. Retrying...")

        wait_time = delay * (2 ** attempt)
        time.sleep(wait_time)
    print("Maximum number of attempts reached. No valid response generated.")
    return None


if __name__ == "__main__":
    key = os.getenv("GEMINI_KEY")
    gemini.configure(api_key = key)
    model = gemini.GenerativeModel("gemini-2.0-flash-lite")

    conversations = [
        'Determinato e Tranquillo', 'Felice e Divertente', 
        'Ambizioso e Incoraggiante', 'Preoccupato e Ottimista', 
        'Amorevole e Rispettoso', 'Fiducioso e Calmo', 
        'Stimolante e Complice', 'Costruttivo e Riflessivo'
        ]
    names = [
        "Alessia", "Matteo", "Giulia", "Andrea", "Chiara", "Marco",
        "Sofia", "Leonardo", "Aurora", "Francesco", "Gaia", "Riccardo",
        "Emma", "Alessandro", "Greta", "Lorenzo", "Anna", "Gabriele",
        "Martina", "Tommaso", "Sara"
        ]

    outputs = []
    for i in tqdm.tqdm(range(NUM_CONVERSATIONS)):

        conversation_type = conversations[(i // 100) % len(conversations)]
        name1, name2 = random.sample(names, 2)
        
        data = {
            "conevrsation_type": conversation_type,
            "name1": name1,
            "name2": name2
        }
        
        prompt = generate_prompt(data)
        response = generate_conversation(model, prompt)
        outputs.append(response)
    
    random.shuffle(outputs)
    with open(".data/processed/safe_conversations.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)