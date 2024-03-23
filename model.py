import torch
import torch.nn as nn
from torch.nn import functional as F
import math

import numpy as np

class RandomRatingGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        """
        Generate a random rating for each input token
        The output is a tensor of shape (batch_size, sequence_length, vocab_size)

        There should be a 1.0 in the position of the generated rating and 0.0 elsewhere
        The rating positions are 1 through 5, and the movie positions are 6 through vocab_size
        """

        batch_size, sequence_length = x.size()
        output = torch.zeros(batch_size, sequence_length, self.vocab_size)
        for i in range(batch_size):
            for j in range(sequence_length):
                rating_position = torch.randint(1, 6, (1,))
                output[i, j, rating_position] = 1.0

        return output, None
    
class AverageRatingGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        """
        Generate rating for the last input token
        The output is a tensor of shape (batch_size, sequence_length, vocab_size)

        There should be a 1.0 in the position of the generated rating and 0.0 elsewhere
        The rating positions are 1 through 5, and the movie positions are 6 through vocab_size

        The rating should be the average of the ratings of the movies in the sequence
        """

        batch_size, sequence_length = x.size()
        output = torch.zeros(batch_size, sequence_length, self.vocab_size)
        for i in range(batch_size):
            ratings = x[i, 2::2]
            average_rating = torch.round(ratings.float().mean())
            output[i, -1, int(average_rating)] = 1.0

        return output, None

class BatchMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.key = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=False)

        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.att_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, nh, T, T)
        wei = self.att_dropout(wei)

        out = wei @ v  # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        out = self.resid_dropout(out)

        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        self.sa = BatchMultiHeadAttention(config)
        self.ffw = FeedFoward(config)

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connections
        x = x + self.ffw(self.ln2(x))  # Residual connections
        return x


class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embed)

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        self.block_size = config.block_size
        self.device = config.device
        self.criteria = config.criteria


    def forward(self, idx, targets=None):
        idx = idx[:, -self.block_size :]
        B, T = idx.shape

        tok_embed = self.token_embedding_table(idx)  # (B, T, n_embed) C being embed size in this case
        pos_embed = self.position_embedding(torch.arange(self.block_size - T, self.block_size, device=self.device))
        
        x = tok_embed + pos_embed  # (B, T, C)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        if self.criteria == "cross_entropy":
            loss = F.cross_entropy(logits, targets, ignore_index=5)
        if self.criteria == "masked_cross_entropy":
            mask = targets > 4
            targets = targets.masked_fill(mask, 5)
            loss = F.cross_entropy(logits, targets, ignore_index=5)
        if self.criteria == "rating_masked_cross_entropy":
            mask = targets > 4
            targets = targets.masked_fill(mask, 5)
            loss = F.cross_entropy(logits[:, :5], targets, ignore_index=5)
        if self.criteria == "distance_weighted_loss":
            mask = targets > 4
            targets = targets.masked_fill(mask, 5)
            loss = custom_distance_weighted_loss_with_mask(logits[:, :5], targets, 5, 5)
        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # Shape is (B, T, C)
            last_logits = logits[:, -1, :]  # Becomes (B, C)
            probs = F.softmax(last_logits, dim=-1)  # (B, C)
            index = torch.multinomial(probs, num_samples=1)  # Becomes (B, 1)
            idx = torch.cat((idx, index), dim=-1)  # Becomes (B, T + 1)

        return idx


def custom_distance_weighted_loss_with_mask(outputs, targets, num_classes, mask_value):
    """
    A custom distance-weighted loss function that incorporates class masking.

    Args:
    - outputs: The raw logits from the model (B, num_classes).
    - targets: The ground truth labels (B,).
    - num_classes: The total number of classes.
    - mask_value: The value used to indicate which targets to mask/ignore.

    Returns:
    - The computed loss.
    """
    # Convert logits to softmax probabilities
    softmax_probs = F.softmax(outputs, dim=-1)
    
    # Compute a mask for targets to be included in loss computation
    mask = targets != mask_value
    masked_targets = targets[mask]
    
    # If no targets are left after masking, return a null loss
    if len(masked_targets) == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Apply mask to softmax probabilities
    masked_softmax_probs = softmax_probs[mask]
    
    # Convert masked targets to one-hot encoding
    target_probs = F.one_hot(masked_targets, num_classes=num_classes).float()
    
    # Compute loss using MSE between weighted softmax probabilities and weighted targets
    loss = torch.mean((masked_softmax_probs - target_probs) ** 2)

    return loss