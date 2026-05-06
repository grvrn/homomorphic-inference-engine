# homomorphic-inference-engine

A privacy-preserving machine learning inference engine that enables secure predictions on encrypted data using the Paillier cryptosystem. This project demonstrates evaluating a linear model (e.g., classifying handwritten MNIST digits 0 vs 1) entirely under encryption.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train the Model](#1-train-the-model)
  - [2. Homomorphic Inference](#2-homomorphic-inference)
  - [3. Benchmarks](#3-benchmarks)
- [How it Works](#how-it-works)

## Features
- **Privacy-Preserving Inference**: Client (Alice) encrypts her data, Server (Carol) computes the linear model blindly, and Alice decrypts the result.
- **Paillier Cryptosystem**: Supports homomorphic addition and scalar multiplication, ideal for linear models.
- **Fixed-Point Arithmetic**: Scales floating-point weights and features into integers to allow encryption.
- **End-to-End Pipeline**: Includes model training, generating HE-compatible specs, encryption, evaluation, and decryption.

## Project Structure

```
homomorphic-inference-engine/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
│
├── src/
│   └── app.py                  # Core application logic and orchestration
│
├── crypto/
│   ├── crypto.py               # Homomorphic encryption/decryption operations
│   └── keys.py                 # Key generation and management (public/secret/relin keys)
│
├── models/
│   ├── train.py                # Script to train models and export HE specs
│   ├── he_spec.py              # HE logic definition and serialization
│   ├── inference.py            # Feature transformation and logit retrieval
│   └── (generated artifacts)   # e.g., classifier.pkl, linear_he_spec.json
│
└── benchmarks/
    ├── accuracy_tests.py       # Accuracy evaluation (encrypted vs. plaintext inference)
    └── latency_tests.py        # Latency and throughput benchmarks
```

### Directory Overview

| Directory | Purpose |
|-----------|---------|
| **`src/`** | Core application logic — wires together encryption, model loading, and inference |
| **`crypto/`** | Homomorphic encryption primitives — key management, encrypt/decrypt routines |
| **`models/`** | ML model training scripts, feature packing, and generating parameters compatible with Homomorphic Encryption |
| **`benchmarks/`** | Performance and correctness tests comparing encrypted vs. plaintext inference |

## Installation

Ensure you have Python 3 installed. It is strongly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install the required dependencies
pip install -r requirements.txt
```

## Usage

### 1. Homomorphic Inference

You can run homomorphic inference directly on a preprocessed image (e.g., an image of a handwritten 0 or 1). 

```bash
python main.py --predict /path/to/image.png
```
The script will load the image, preprocess it, encrypt features using generated Paillier keys, run an encrypted linear evaluation, decrypt the score, and compare it with the plaintext prediction.

### 2. Benchmarks

The project includes built-in benchmarking for both **latency** and **accuracy**.

**To evaluate accuracy (homomorphic vs plaintext):**
```bash
python main.py --test accuracy --he-samples 20
```

**To evaluate latency scaling over the number of features:**
```bash
python main.py --test latency
```
*(This will generate a `latency_comparison.png` plot comparing computation time).*

## How it Works

1. **Client (Alice)**: Converts raw features (e.g., image pixels) to scaled integer values based on the generated HE spec, encrypts them using her Paillier public key, and sends the ciphertexts to the server.
2. **Server (Carol)**: Loads the exported model weights (also converted to scaled integers), computes the dot product using homomorphic addition and scalar multiplication, and sends the encrypted result back to Alice.
3. **Client (Alice)**: Decrypts the result using her secret key, reverses the scaling, and determines the final predicted logit/label.
