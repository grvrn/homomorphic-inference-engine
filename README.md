# homomorphic-inference-engine

A privacy-preserving machine learning inference engine

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
│   ├── poly_approx.py          # Polynomial approximations for non-linear activation functions
│   └── classifier.pkl          # Serialized (pre-trained) classifier model
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
| **`models/`** | ML model artifacts and polynomial approximations for HE-compatible activation functions |
| **`benchmarks/`** | Performance and correctness tests comparing encrypted vs. plaintext inference |
