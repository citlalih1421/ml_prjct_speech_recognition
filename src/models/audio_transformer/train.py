import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer
from model import AudioTransformerModel
from dataset import AudioDataset  

def train_model(train_dataset, model, tokenizer, output_dir='./results'):
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # Save the model after training
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    # Initialize the dataset and dataloaders
    train_dataset = AudioDataset(
        base_dir='ml_prjct_speech_recognition/data/processed/common-voice/cv-valid-train', 
        tokenizer=tokenizer,
        target_seq_len=128
    )

    # Initialize the model
    model = AudioTransformerModel()

    # Train the model
    train_model(train_dataset, model, tokenizer)
