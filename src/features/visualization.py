import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cm(cm, class_labels, name):
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
    plt.savefig(f"./reports/figures/{name}.png")
    plt.show()
    
def plot_loss(trainer):
    log_history = trainer.state.log_history
    
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    
    for log in log_history:
        if 'loss' in log and 'eval_loss' not in log:
            train_losses.append(log['loss'])
            train_steps.append(log['step'])
        elif 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])
            eval_steps.append(log['step'])
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_steps, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(eval_steps, eval_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("./reports/figures/gilBERTo_loss.png")
    plt.show()