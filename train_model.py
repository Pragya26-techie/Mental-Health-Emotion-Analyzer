import pandas as pd
import transformers
import accelerate
import datasets
print(transformers.__version__)
print(accelerate.__version__)
print(datasets.__version__)
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch



if __name__ == "__main__":
    # Load and prepare data
    train_df = pd.read_csv(r'D:\Mental-Health-Emotion-Analyzer\Data\train.csv')
    test_df = pd.read_csv(r'D:\Mental-Health-Emotion-Analyzer\Data\test.csv')
    valid_df = pd.read_csv(r'D:\Mental-Health-Emotion-Analyzer\Data\valid.csv')

    # Label mapping
    label2id = {label: idx for idx, label in enumerate(train_df['emotion'].unique())}
    train_df["label"] = train_df["emotion"].map(label2id)
    test_df["label"] = test_df["emotion"].map(label2id)
    valid_df["label"] = valid_df["emotion"].map(label2id)

    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # Tokenize
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    valid_dataset = valid_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text", "emotion"])
    test_dataset = test_dataset.remove_columns(["text", "emotion"])
    valid_dataset = valid_dataset.remove_columns(["text", "emotion"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")
    valid_dataset.set_format("torch")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label2id))

    # Training args
    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )
    print("TrainingArguments loaded successfully ✅")

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    #  Start training
    trainer.train()
    trainer.save_model("saved_model")  # Saves model + tokenizer config
    tokenizer.save_pretrained("saved_model")


    
    



 