import os
import torch
import logging
import librosa
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from torch.nn.utils.rnn import pad_sequence
import evaluate


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load processor
logger.info("Loading processor...")
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Paths
csv_train_path = os.path.expanduser("~/ml_prjct_speech_recognition/data/raw/common-voice/cv-valid-train.csv")
csv_test_path = os.path.expanduser("~/ml_prjct_speech_recognition/data/raw/common-voice/cv-valid-test.csv")
train_audio_dir = os.path.expanduser("~/ml_prjct_speech_recognition/data/raw/common-voice")
test_audio_dir = os.path.expanduser("~/ml_prjct_speech_recognition/data/raw/common-voice")

# Load dataset
logger.info("Loading dataset...")
data_files = {"train": csv_train_path, "test": csv_test_path}
dataset = load_dataset("csv", data_files=data_files)

# Select subset for faster testing (optional)
logger.info("Selecting a subset of the dataset for testing...")
dataset["train"] = dataset["train"].select(range(5000)) 
dataset["test"] = dataset["test"].select(range(3995)) 
logger.info("Subset selected: train = 1000 samples, test = 100 samples")

# Data collator
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_values = [torch.tensor(feature["input_values"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]
        input_values = pad_sequence(input_values, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_values": input_values, "labels": labels}

# Preprocess function
def preprocess(batch):
    file_path = os.path.join(train_audio_dir if "train" in batch["filename"] else test_audio_dir, batch["filename"])
    speech, _ = librosa.load(file_path, sr=16000)
    batch["input_values"] = processor(speech, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

logger.info("Preprocessing train dataset...")
dataset["train"] = dataset["train"].map(preprocess, desc="Preprocessing Train Dataset")
logger.info("Train dataset preprocessing complete.")

logger.info("Preprocessing test dataset...")
dataset["test"] = dataset["test"].map(preprocess, desc="Preprocessing Test Dataset")
logger.info("Test dataset preprocessing complete.")

# Model
logger.info("Loading model...")
model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
logger.info("Model loaded successfully.")

# Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned",
    group_by_length=True,
    per_device_train_batch_size=4,
    eval_strategy="steps",
    logging_dir="./logs", 
    logging_steps=50, 
    num_train_epochs=5,
    save_steps=500,
    eval_steps=500,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
    fp16=False,
    no_cuda=True,
)

# Metrics
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


# Trainer
logger.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    data_collator=CustomDataCollator(processor),
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=processor,
)
logger.info("Trainer initialized.")

# Train
logger.info("Starting training...")
trainer.train()
logger.info("Training complete.")

# Evaluate and save
logger.info("Starting evaluation...")
results = trainer.evaluate()
logger.info(f"Word Error Rate (WER): {results['eval_wer']}")
model.save_pretrained("./wav2vec2-finetuned")
processor.save_pretrained("./wav2vec2-finetuned")
logger.info("Model and processor saved.")
