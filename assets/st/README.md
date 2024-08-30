# ONNX Model Directory

This directory contains the ONNX model and associated configuration files for the `sentence-transformers/all-MiniLM-L6-v2` model, which is based on the `BertModel` architecture. These files are used to deploy the model in applications that require text embeddings or similar NLP tasks.

## Files Description

- **config.json**: Contains the configuration settings for the model, specifying model architecture and other parameters necessary for initialization.
- **model.onnx**: The ONNX (Open Neural Network Exchange) format of the model, which allows for model use in various frameworks that support ONNX.
- **special_tokens_map.json**: Provides a mapping for special tokens like [CLS], [SEP], etc., that are used within the tokenizer.
- **tokenizer.json**: Configuration file for the tokenizer, detailing how text should be broken down during preprocessing.
- **vocab.json**: A dictionary file mapping tokens to their respective indices in the vocabulary.
- **vocab.txt**: A plaintext file listing the vocabulary tokens used by the tokenizer, one per line.

## Model Details

- **Model Name**: `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture**: `BertModel