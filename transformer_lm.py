# models.py

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
import math


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()

        self.query_layer = nn.Linear(d_model, d_internal)
        self.key_layer = nn.Linear(d_model, d_internal)
        self.value_layer = nn.Linear(d_model, d_internal)
        self.output_projection = nn.Linear(d_internal, d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )

        self.d_model = d_model

    def forward(self, input_vecs):
        queries = self.query_layer(input_vecs)
        keys = self.key_layer(input_vecs)
        values = self.value_layer(input_vecs)

        keys_t = keys.transpose(-2, -1)

        scores = torch.matmul(queries, keys_t) / np.sqrt(self.d_model)
        attention = torch.softmax(scores, dim=0)
        attention = torch.matmul(attention, values)

        attention_output = self.output_projection(attention)

        attention_output = input_vecs + attention_output

        ff_output = self.feedforward(attention_output)

        output = attention_output + ff_output

        return output, attention


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, d_internal, chunk_size):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        # Use TransformerEncoderLayer with nn.TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_internal)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to map to vocabulary size
        self.linear = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_indices, causal_mask):
        # Embed and apply positional encoding
        x = self.embedding(input_indices)
        x = self.positional_encoding(x)

        #x = x.transpose(0, 1)

        # Apply the transformer encoder with the causal mask
        x = self.transformer_encoder(x,
                                     mask=causal_mask)  # transpose to shape [seq_len, batch, d_model]
        x = x.transpose(0, 1)  # transpose back to [batch, seq_len, d_model]

        logits = self.linear(x)
        return self.log_softmax(logits), None  # No attention maps to return here

    def create_causal_mask(self, seq_len):
        # Create an upper triangular matrix with -inf for causal masking
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, d_model, num_layers, d_internal, trained_state_dict, vocab_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Save the vocab index for character-to-index conversions
        self.vocab_index = vocab_index

        # Load the trained weights from the state_dict
        self.load_trained_weights(trained_state_dict)

    def load_trained_weights(self, trained_state_dict):
        """
        Loads the trained weights into the model.
        """
        self.embedding.load_state_dict(trained_state_dict['embedding'])
        for i, layer in enumerate(self.transformer_layers):
            layer.load_state_dict(trained_state_dict[f'transformer_layers.{i}'])
        self.linear.load_state_dict(trained_state_dict['linear'])

    def forward(self, input_indices, causal_mask):
        # Embed and apply positional encoding
        x = self.embedding(input_indices)
        x = self.positional_encoding(x)

        attention_maps = []
        for layer in self.transformer_layers:
            x, attention_map = layer(x, causal_mask)
            attention_maps.append(attention_map)

        logits = self.linear(x)
        return self.log_softmax(logits), attention_maps

    def get_next_char_log_probs(self, context):
        """
        Gets the log probabilities for the next character given a context.
        :param context: The string context.
        :return: Log probability distribution over the vocabulary as a numpy array.
        """
        self.eval()  # Ensure dropout is disabled during inference

        # Convert context into indices using vocab_index
        context_indices = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0)
        causal_mask = self.create_causal_mask(context_indices.size(1))

        # Forward pass through the Transformer
        log_probs, _ = self.forward(context_indices, causal_mask)

        # Return log probabilities for the last character
        return log_probs[0, -1].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a sequence of next characters following the given context.
        :param next_chars: Sequence of characters to score.
        :param context: The string context.
        :return: Log probability of the sequence.
        """
        log_prob = 0.0
        for i in range(len(next_chars)):
            log_probs = self.get_next_char_log_probs(context + next_chars[:i])
            log_prob += log_probs[self.vocab_index.index_of(next_chars[i])]
        return log_prob

    def create_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    vocab_size = len(vocab_index)
    chunk_size = 500  # You can adjust this based on your memory

    # Initialize the model
    model = TransformerLM(vocab_size=vocab_size, d_model=128, num_layers=3, d_internal=64, chunk_size=chunk_size)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()

    # Prepare the training data (chunks)
    train_data = [train_text[i:i + chunk_size - 1] for i in range(0, len(train_text) - chunk_size)]
    # Add a space at the beginning of each chunk
    train_data = [' ' + chunk for chunk in train_data]

    num_epochs = 1
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i in range(len(train_data)):

            if (i % 100 == 0):
                print(f'Iteration {i} | Remaining: {len(train_data) - i}')

            chunk = train_data[i]
            input_chars = torch.tensor([vocab_index.index_of(c) for c in chunk[:-1]], dtype=torch.long)
            target_chars = torch.tensor([vocab_index.index_of(c) for c in chunk[1:]], dtype=torch.long)

            input_chars = input_chars.unsqueeze(1)

            # Create causal mask
            seq_len = input_chars.size(0)
            causal_mask = model.create_causal_mask(seq_len)

            # Forward pass
            optimizer.zero_grad()
            log_probs, _ = model(input_chars, causal_mask)

            # Compute the loss
            preds = log_probs.view(-1, vocab_size)
            truth = target_chars.view(-1)
            loss = loss_fcn(preds, truth)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}\n")

    # Save the model's state_dict (weights) after training
    trained_state_dict = model.state_dict()
    model.eval()
    return trained_state_dict, model
