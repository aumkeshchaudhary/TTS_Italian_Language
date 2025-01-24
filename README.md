# Fine-tuning SpeechT5 TTS Model for Italian

                  Aumkeshchy2003/Italian_TTS

This repository provides a comprehensive guide to fine-tuning the SpeechT5 Text-to-Speech (TTS) model on Italian language datasets. The goal is to adapt Microsoft’s SpeechT5 model to generate Italian speech from text, leveraging the power of the Hugging Face Transformers library.
Table of Contents

    Introduction
    Requirements
    Dataset Preparation
    Model Fine-tuning
    Usage
    Results
    References

Introduction

SpeechT5 is a powerful model designed by Microsoft for Text-to-Speech tasks. It has demonstrated superior performance in many languages, but here, we aim to fine-tune it specifically for Italian. This README explains the process of preparing data, setting up the environment, fine-tuning the model, and generating Italian speech.
Requirements

Before getting started, make sure you have the following dependencies installed:

    Python 3.7+
    PyTorch >= 1.9.0
    Transformers >= 4.28.0
    Datasets
    torchaudio
    librosa
    numpy
    gradio (for inference demo)

You can install the required packages by running:

    pip install torch transformers datasets torchaudio librosa gradio

**Dataset Preparation**

For fine-tuning the TTS model, we need an Italian language dataset. Ensure that your dataset is structured in a format compatible with Hugging Face datasets library.

Each data point should include:

    text: The input Italian text to be spoken.
    audio: The corresponding speech waveform.

You can preprocess your dataset like this:

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

**Model Fine-tuning**
***Steps***

    1. Initialize the model and tokenizer:

     from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
     import torch
     # Load the pre-trained SpeechT5 model and processor
     model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
     processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

   2. Prepare for fine-tuning:
    
    Make sure your dataset and model inputs are correctly processed.

   3. Train the model:

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

   4. Save the model:

    After fine-tuning, save your model for later inference.

    model.save_pretrained("./fine_tuned_speecht5_italian")
    processor.save_pretrained("./fine_tuned_speecht5_italian")

**Usage**

Once the model is fine-tuned, you can generate Italian speech from any text.

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

**Results**

After fine-tuning, you should be able to generate realistic Italian speech. Further improvements can be achieved by experimenting with the dataset, hyperparameters, and training epochs.
References

   * SpeechT5 on Hugging Face
   * Transformers Documentation

