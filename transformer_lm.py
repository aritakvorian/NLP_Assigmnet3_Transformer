# models.py

import numpy as np
import torch
from torch import nn
import math

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_head, num_layers, d_internal, chunk_size, dropout=0.1):
        super(TransformerLM, self).__init__()

        self.d_model = d_model
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, chunk_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, dim_feedforward=d_internal,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input, src_mask):
        embedded_input = self.embedding(input)
        embedded_input = embedded_input * math.sqrt(self.d_model)

        encoded_input = self.positional_encoding(embedded_input)

        output = self.transformer_encoder(encoded_input, src_mask)
        output = self.fc_out(output)
        return nn.functional.log_softmax(output, dim=-1)

    def generate(self, input):

        embedded_input = self.embedding(input)
        embedded_input = embedded_input * math.sqrt(self.d_model)

        encoded_input = self.positional_encoding(embedded_input)

        output = self.transformer_encoder(encoded_input)
        output = self.fc_out(output)
        return nn.functional.log_softmax(output, dim=-1)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer_lm, vocab_index):
        self.transformer_lm = transformer_lm
        self.vocab_index = vocab_index
        self.vocab_size = len(vocab_index)

    def get_next_char_log_probs(self, context):
        context_indices = [self.vocab_index.index_of(char) for char in context]
        context_tensor = torch.tensor(context_indices).unsqueeze(0)
        #context_mask = masking(context_tensor.size(1))

        with torch.no_grad():
            self.transformer_lm.eval()
            output = self.transformer_lm.generate(context_tensor)

        return output[0, -1, :].numpy()

    def get_log_prob_sequence(self, next_chars, context):

        if context == "":
            context = " "

        total_log_prob = 0.0

        chunk_size = self.transformer_lm.chunk_size

        context_chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]

        current_chunk = context_chunks[-1]

        for char in next_chars:

            log_probs = self.get_next_char_log_probs(current_chunk)
            char_index = self.vocab_index.index_of(char)
            total_log_prob += log_probs[char_index]

            current_chunk += char

            if len(current_chunk) > chunk_size:
                current_chunk = current_chunk[-chunk_size:]

        return total_log_prob


def masking(size):
    mask = torch.triu(torch.zeros(size, size) + float('-inf'), diagonal=1)
    return mask


def create_chunks(text, chunk_size):
    words = text.split(' ')
    input_chunks = []
    target_chunks = []

    for i in range(len(words) - 1):

        chunk = ' '.join(words[i:])[:chunk_size]

        # Padding
        if len(chunk) < chunk_size:
            chunk = ' ' * (chunk_size - len(chunk)) + chunk

        input_chunk = ' ' + chunk[:chunk_size - 1]
        target_chunk = chunk[:chunk_size]

        input_chunks.append(input_chunk)
        target_chunks.append(target_chunk)

    return input_chunks, target_chunks


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    # Hyperparameters
    chunk_size = 20
    num_epochs = 10
    d_model = 128
    d_internal = 512
    learning_rate = 0.0001
    num_head = 8
    num_layers = 6
    vocab_size = len(vocab_index)

    model = TransformerLM(vocab_size=vocab_size,
                          d_model=d_model,
                          num_head=num_head,
                          num_layers=num_layers,
                          d_internal=d_internal,
                          chunk_size=chunk_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    input_chunks, target_chunks = create_chunks(train_text, chunk_size)

    print(f'Total training chunks: {len(input_chunks)}')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_log_prob = 0
        total_characters = 0

        i = 0

        for input_chunk, target_chunk in zip(input_chunks, target_chunks):

            if i % 1000 == 0 and i > 0:
                print(f'Processing chunk: {i}')

            # Batching option
            input_tensor = torch.tensor([vocab_index.index_of(c) for c in input_chunk]).unsqueeze(0)  # [batch, chunk_size]
            target_tensor = torch.tensor([vocab_index.index_of(c) for c in target_chunk]).unsqueeze(0)  # [batch, chunk_size]

            # No batching
            input_tensor = torch.tensor([vocab_index.index_of(c) for c in input_chunk])
            target_tensor = torch.tensor([vocab_index.index_of(c) for c in target_chunk])

            # Masking
            mask = masking(chunk_size)

            # Forward pass
            optimizer.zero_grad()
            prediction = model(input_tensor, mask)

            test = torch.exp(prediction)

            pred_output = prediction.view(-1, vocab_size)
            truth = target_tensor.view(-1)

            loss = loss_fn(pred_output, truth)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Perplexity calc
            temp = target_tensor.numel()
            total_log_prob += -loss.item() * temp
            total_characters += target_tensor.numel()

            i += 1

        # Outside epoch total perplexity
        avg_log_prob = total_log_prob / total_characters
        perplexity = np.exp(-avg_log_prob)

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss:.4f} | Perplexity: {perplexity:.4f}")

    return NeuralLanguageModel(model, vocab_index)