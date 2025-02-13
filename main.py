import os
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from torchvision import transforms
from datasets import Dataset
import evaluate

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset directories containing images
DATA_DIRS = {
    "train": r"D:\\Soy Vitou\\GITHUB\\DEEP-LEARNING\\TEXT-RECOGNITION\\DATASET-TESTING\\split_dataset\\train",
    "valid": r"D:\\Soy Vitou\\GITHUB\\DEEP-LEARNING\\TEXT-RECOGNITION\\DATASET-TESTING\\split_dataset\\valid",
}

# Text annotation files
ANNOTATION_FILES = {
    "train": r"D:\\Soy Vitou\\GITHUB\\DEEP-LEARNING\\TEXT-RECOGNITION\\DATASET-TESTING\\split_dataset\\train.txt",
    "valid": r"D:\\Soy Vitou\\GITHUB\\DEEP-LEARNING\\TEXT-RECOGNITION\\DATASET-TESTING\\split_dataset\\valid.txt",
}

# File containing all unique characters
CHAR_FILE = r"D:\\Soy Vitou\\GITHUB\\DEEP-LEARNING\\TEXT-RECOGNITION\\DATASET-TESTING\\annotation.txt"


def get_unique_chars(file_path):
    """Extracts unique Khmer characters from the annotation file."""
    unique_chars = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                unique_chars.update(parts[1])
    return sorted(unique_chars)


khmer_tokens = get_unique_chars(CHAR_FILE)
print(f"Total unique Khmer characters: {len(khmer_tokens)}")

# Load the TrOCR processor (pretrained model for OCR)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Add new Khmer tokens to the tokenizer
new_token_count = processor.tokenizer.add_tokens(khmer_tokens)
print(f"Added {new_token_count} new tokens to tokenizer")

# Load TrOCR model
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

# Resize embeddings to include new tokens
if new_token_count > 0:
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Image preprocessing transform
image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_dataset(data_dir, annotation_file):
    """Loads images and labels into a Hugging Face Dataset."""
    image_paths, texts = [], []
    with open(annotation_file, encoding="utf-8") as f:
        for line in f:
            image_name, text = line.strip().split(" ", 1)
            image_path = os.path.join(data_dir, image_name)
            if os.path.exists(image_path):
                image_paths.append(image_path)
                texts.append(text)
    return Dataset.from_dict({"image": image_paths, "text": texts})


def preprocess_data(example):
    image = Image.open(example["image"]).convert("RGB")
    example["pixel_values"] = image_transform(image)
    example["labels"] = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=128,
        truncation=True
    ).input_ids
    return example


# Load datasets
train_dataset = load_dataset(DATA_DIRS["train"], ANNOTATION_FILES["train"])
valid_dataset = load_dataset(DATA_DIRS["valid"], ANNOTATION_FILES["valid"])

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_data)
valid_dataset = valid_dataset.map(preprocess_data)

# Metric for evaluation
cer_metric = evaluate.load("cer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = processor.batch_decode(np.argmax(logits, axis=-1), skip_special_tokens=True)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    label_texts = processor.batch_decode(labels, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=predictions, references=label_texts)
    return {"cer": cer}


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-khmer",
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=200,
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
