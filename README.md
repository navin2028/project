# project
# Character-Level RNN Language Model

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Text Generation](#text-generation)
  - [Evaluating Perplexity](#evaluating-perplexity)
  - [Plotting Training Loss](#plotting-training-loss)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a simple character-level Recurrent Neural Network (RNN) using PyTorch. The model is trained on a small dataset of text sequences and can be used to generate text character by character. It includes features like one-hot encoding, training, text generation, and evaluation using perplexity.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/char-level-rnn.git
    cd char-level-rnn
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install torch numpy matplotlib
    ```

## Usage

### Training

To train the RNN model on your dataset, you can modify the `text` list in the script with your own sequences of text. The model is trained using the following code:

```python
#Text Generation
generated_text = sample(model, out_len=15, start='good')
print(generated_text)  # Example output: 'good i am fine '
#Evaluating Perplexity
import math

with torch.no_grad():
    model.eval()
    output, _ = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())

perplexity = math.exp(loss.item())
print("Perplexity:", perplexity)  # Example output: Perplexity: 1.2016
#Plotting Training Loss
import matplotlib.pyplot as plt

losses = []
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(range(1, n_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

#Install these dependencies using:
pip install torch numpy matplotlib
License.....

### Explanation:
- **Overview:** Provides a brief introduction to the project, including its purpose.
- **Installation:** Instructions on how to set up the environment and install dependencies.
- **Usage:** Details on how to train the model, generate text, evaluate it, and plot training loss.
- **Configuration:** Explains the key parameters that can be configured.
- **Dependencies:** Lists the required libraries and how to install them.
- **Contributing:** Guidelines for contributing to the project.
- **License:** Information about the project's licensing.




