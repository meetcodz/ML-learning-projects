# Character-Level RNN Language Model

A character-level language model built completely from scratch using **PyTorch**. The model learns patterns in text one character at a time and generates new text by predicting the next character based on the sequence it has seen.

This project was built as part of my deep learning journey to understand the fundamentals of sequence modeling before moving on to LSTMs, GRUs, Transformers, and Large Language Models.

---

## Features

* Character-level text generation
* Vanilla Recurrent Neural Network (RNN)
* Implemented using PyTorch
* Custom vocabulary creation
* Sequence generation with hidden state propagation
* Training loop with gradient descent and backpropagation through time
* Temperature-based text sampling

---

## Project Structure

```
Character-Level-RNN/
│
├── Character_Level_RNN.ipynb   # Complete implementation
├── input.txt                   # Training dataset
├── README.md
└── model.pth                   # Saved model (after training)
```

---

## How It Works

### 1. Data Preparation

* Read raw text
* Build a character vocabulary
* Assign every character a unique integer
* Convert text into numerical sequences

Example:

```
Text:
hello

Vocabulary:
['h', 'e', 'l', 'o']

Encoding:
h → 0
e → 1
l → 2
o → 3
```

---

### 2. Creating Training Sequences

Input sequences consist of fixed-length character windows.

Example:

```
Input:
hell

Target:
ello
```

The model learns to predict each next character.

---

### 3. Model Architecture

```
Input Characters
        │
        ▼
Character Embeddings
        │
        ▼
Vanilla RNN
        │
        ▼
Linear Layer
        │
        ▼
Softmax
        │
        ▼
Next Character Prediction
```

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Google Colab

---

## Training

The model is trained using:

* Cross Entropy Loss
* Adam Optimizer
* Backpropagation Through Time (BPTT)

Training process:

```
Input Sequence
      ↓
Forward Pass
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Parameter Update
```

---

## Text Generation

After training, the model generates text one character at a time.

Generation process:

1. Provide a starting prompt.
2. Predict probability distribution for the next character.
3. Sample one character.
4. Feed it back into the model.
5. Repeat until the desired length is reached.

---

## Example Output

```
Input Prompt:
The

Generated Text:
The night was calm and the stars shimmered above the silent forest...
```

*(Generated text quality depends on the dataset size, training time, and model configuration.)*

---

## Concepts Learned

Through this project I gained hands-on experience with:

* Recurrent Neural Networks
* Hidden States
* Sequence Modeling
* Character Encoding
* Cross Entropy Loss
* Backpropagation Through Time
* Teacher Forcing
* Text Sampling
* Temperature Sampling
* Language Modeling

---

## Future Improvements

* Replace vanilla RNN with LSTM
* Implement GRU
* Train on a larger corpus
* Add beam search decoding
* Train using mini-batches
* Add checkpoint saving and loading
* Build a word-level language model
* Extend the project to a Transformer-based language model

---

## Results

The trained model successfully learns character-level patterns in the dataset and is capable of generating coherent text after sufficient training. While limited by the capacity of a vanilla RNN, this project provides a strong foundation for understanding modern sequence models.

---

## Learning Goal

The purpose of this project is educational. Instead of using pre-built language models, the objective was to understand every stage of building a language model—from preprocessing and sequence creation to training, backpropagation, and autoregressive text generation.

---

## License

This project is open source and available under the MIT License.
