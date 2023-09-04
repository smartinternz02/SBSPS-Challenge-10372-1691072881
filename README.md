# SBSPS-Challenge-10372-1691072881
Media Monitoring Multilabel Classification :Multi-label classification of printed media articles to topics

# Project Readme: RCNN Model for Text Classification

## Introduction

Welcome to the README file for our project implementing a Recurrent Convolutional Neural Network (RCNN) model for text classification. This project is inspired by the paper "Recurrent Convolutional Neural Networks for Text Classification."

## Problem Statement

Traditional Recurrent Neural Networks (RNNs) are effective in text classification because they capture contextual information while representing a sentence as a fixed-size vector. However, these representations can be biased, giving more importance to words appearing later in the sentence, potentially causing the loss of crucial semantic information from the beginning of the sentence.

Conversely, Convolutional Neural Networks (CNNs) do not suffer from this issue, as their (max) pooling layers identify discriminative phrases in the text. Nonetheless, traditional CNNs may struggle to capture the full semantics of the text due to their simplistic convolutional filters.

The RCNN model presented here combines both RNN and CNN architectures to better capture the semantics of text. It first uses a recurrent structure to learn word representations, preserving contextual information and accommodating various word orderings. Then, max-pooling is applied to identify key features relevant to the classification task. By uniting these two approaches, RCNN leverages the strengths of both recurrent and convolutional neural models.

## Model Architecture

The RCNN architecture can be summarized as follows:

1. **Bi-directional RNN:** A bi-directional RNN is employed to identify left (l_i) and right (r_i) context for each word w_i. These context vectors are concatenated to represent a word:
   ```
   x_i = [l_i ; e_i ; r_i]
   ```

2. **Word Representations:** These word representations are then passed through a linear layer followed by a tanh activation function to obtain new representations in dimension h.

3. **Max-Pooling:** Max-pooling is performed on the h-dimensional vectors (elementwise) to generate the final h-dimensional feature map.

4. **Classification Layer:** The feature map is passed through a linear layer followed by a softmax activation function for classification.

## Implementation Details

Here are some key implementation details for our RCNN model:

- **Word Embeddings:** We used pre-trained Glove embeddings for initializing word vectors, providing a strong foundation for word representations.

- **BiLSTM Hidden Units:** We employed a bidirectional LSTM (BiLSTM) with 64 hidden units to capture context effectively.

- **Dropout:** Dropout with a keep probability of 0.8 was applied to prevent overfitting during training.

- **Optimizer:** Stochastic Gradient Descent (SGD) was used as the optimizer to update model parameters.

- **Loss Function:** We utilized the Negative Log Likelihood loss function for training the model.
