from features.preprocessing import conversation_loader
from features.preprocessing import text_processing
from models.baseline import Baseline

texts, labels = conversation_loader()
baseline = Baseline()
baseline.train(texts, labels)

test = "Non essere così critico, è normale avere dei dubbi. Non dovresti essere troppo severo con te stesso. Sono certa che sia colpa mia. Non so cosa non ho fatto per meritare il tuo amore e il tuo rispetto."
test = text_processing(test)
pred_label, pred_proba = baseline.inference([test])
print(f"Label predetta: {pred_label}")
print(f"Probabilità: {pred_proba}")