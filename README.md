# 🎭 Fine-tuning SpeechT5 TTS Model for Italian

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/Italian_TTS)

Transform Italian text into natural speech with our fine-tuned SpeechT5 model. This repository provides a comprehensive guide to adapting Microsoft's SpeechT5 model for Italian Text-to-Speech (TTS), utilizing the Hugging Face Transformers library.

## 📑 Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Model Fine-tuning](#model-fine-tuning)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## 🎯 Introduction

SpeechT5, Microsoft's powerful Text-to-Speech model, has shown remarkable performance across various languages. This project focuses on fine-tuning it specifically for Italian, providing you with the tools and instructions needed to create high-quality Italian synthetic speech.

## 🛠️ Requirements

### Prerequisites
- Python 3.7+
- PyTorch >= 1.9.0
- Transformers >= 4.28.0
- Datasets
- torchaudio
- librosa
- numpy
- gradio (for inference demo)

### Installation

```bash
pip install torch transformers datasets torchaudio librosa gradio
```

## 📊 Dataset Preparation

Prepare your Italian dataset in a format compatible with the Hugging Face datasets library. Each data point should contain:
- `text`: Italian text input
- `audio`: Corresponding speech waveform

Here's how to preprocess your dataset:

```python
import librosa
from datasets import Dataset

# Example: Loading an Italian dataset
dataset = Dataset.from_dict({
    "text": ["Ciao, come stai?", "Buongiorno a tutti!"],
    "audio": ["path_to_audio_file1.wav", "path_to_audio_file2.wav"]
})

# Load the audio
def load_audio(batch):
    speech_array, sampling_rate = librosa.load(batch["audio"], sr=16000)
    batch["speech_array"] = speech_array
    batch["sampling_rate"] = sampling_rate
    return batch

dataset = dataset.map(load_audio)
```

## 🔄 Model Fine-tuning

### Step 1: Initialize Model and Tokenizer

```python
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
import torch

# Load the pre-trained SpeechT5 model and processor
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
```

### Step 2: Train the Model

```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    save_steps=500,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()
```

### Step 3: Save the Model

```python
model.save_pretrained("./fine_tuned_speecht5_italian")
processor.save_pretrained("./fine_tuned_speecht5_italian")
```

## 💫 Usage

Generate Italian speech using your fine-tuned model:

```python
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
import gradio as gr

# Load fine-tuned model
model = SpeechT5ForTextToSpeech.from_pretrained("./fine_tuned_speecht5_italian")
processor = SpeechT5Processor.from_pretrained("./fine_tuned_speecht5_italian")

def tts_infer(text):
    inputs = processor(text, return_tensors="pt")
    speech = model.generate(**inputs)
    return speech

# Run demo using Gradio
demo = gr.Interface(fn=tts_infer, inputs="text", outputs="audio")
demo.launch()
```

## 📊 Results

After fine-tuning, you'll be able to generate natural-sounding Italian speech. Model performance can be further enhanced by:
- Experimenting with different hyperparameters
- Adjusting the number of training epochs
- Using larger or more diverse datasets

## 📚 References

- [SpeechT5 on Hugging Face](https://huggingface.co/microsoft/speecht5_tts)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)

---

💡 Try out the model on our [Hugging Face Space](https://huggingface.co/spaces/Aumkeshchy2003/Italian_TTS)!
