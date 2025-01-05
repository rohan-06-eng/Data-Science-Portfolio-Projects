# Next Word Prediction using LSTM RNN

This project demonstrates the implementation of a Next Word Prediction model using Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN). The model is trained to predict the next word in a given sequence of words.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
Next Word Prediction is a common task in Natural Language Processing (NLP) where the goal is to predict the next word in a sequence given the previous words. This project uses an LSTM RNN to achieve this task.

## Dataset
The dataset used for training the model is a text corpus. It can be any large text dataset, such as books, articles, or scraped web content. The dataset is preprocessed to create sequences of words for training the model.

## Model Architecture
The model is built using LSTM layers, which are well-suited for sequence prediction tasks due to their ability to capture long-term dependencies. The architecture includes:
- An embedding layer to convert words into dense vectors.
- One or more LSTM layers to process the sequences.
- A dense layer with a softmax activation function to output the probability distribution of the next word.

## Training
The model is trained using the following steps:
1. Preprocess the text data to create input-output pairs.
2. Tokenize the text and convert words to integer sequences.
3. Pad sequences to ensure uniform input length.
4. Train the model using a suitable optimizer and loss function.

## Evaluation
The model's performance is evaluated using metrics such as accuracy and perplexity. The evaluation is done on a separate validation dataset to ensure the model generalizes well to unseen data.

## Usage
To use the model for next word prediction:
1. Load the trained model.
2. Provide a sequence of words as input.
3. The model will output the predicted next word.

## Results
The model achieves satisfactory results in predicting the next word in a sequence. The performance can be further improved by using a larger dataset and fine-tuning the model parameters.

## Conclusion
This project demonstrates the effectiveness of LSTM RNNs in the task of next word prediction. The model can be used in various applications such as text autocompletion and predictive text input.

## References
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence Prediction with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/sequence-prediction-lstm-recurrent-neural-networks-python-keras/)
- [Natural Language Processing with Python](https://www.nltk.org/book/)

## Output
![Output Image](https://github.com/rohan-06-eng/Data-Science-Portfolio-Projects/blob/main/Next%20Word%20Prediction%20using%20LSTM%20RNN/output/1.png)
![Continued Output Image](https://github.com/rohan-06-eng/Data-Science-Portfolio-Projects/blob/main/Next%20Word%20Prediction%20using%20LSTM%20RNN/output/2.png)
![Continued Output Image](https://github.com/rohan-06-eng/Data-Science-Portfolio-Projects/blob/main/Next%20Word%20Prediction%20using%20LSTM%20RNN/output/3.png)