from zipfile import ZipFile
from urllib.request import urlretrieve
import os

# Third-party imports
import pandas as pd
import torch
from torch.nn import functional as F
from model import RandomRatingGenerator
import numpy as np
from typing import List


import matplotlib.pyplot as plt

# Constants for data processing
USER_DATA_FILE = "ml-1m/users.dat"
RATINGS_DATA_FILE = "ml-1m/ratings.dat"
MOVIES_DATA_FILE = "ml-1m/movies.dat"
DATA_SEPARATOR = "::"
DATA_ENCODING = "ISO-8859-1"
TEST_DATA_RATIO = 0.1

# Load and prepare the dataset
def load_data():

    if not all(
        map(lambda f: os.path.exists(f), [USER_DATA_FILE, RATINGS_DATA_FILE, MOVIES_DATA_FILE])
    ):
        urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
        ZipFile("movielens.zip", "r").extractall()
        os.remove("movielens.zip")
        
    users = pd.read_csv(
        USER_DATA_FILE,
        sep=DATA_SEPARATOR,
        names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        encoding=DATA_ENCODING,
        engine="python",
    )

    ratings = pd.read_csv(
        RATINGS_DATA_FILE,
        sep=DATA_SEPARATOR,
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        encoding=DATA_ENCODING,
        engine="python",
    )

    movies = pd.read_csv(
        MOVIES_DATA_FILE,
        sep=DATA_SEPARATOR,
        names=["movie_id", "title", "genres"],
        encoding=DATA_ENCODING,
        engine="python",
    )

    return users, ratings, movies

def ratings_to_sequences(ratings):
    # Sort ratings by user_id and unix_timestamp
    ratings.sort_values(by=["user_id", "unix_timestamp"], inplace=True)

    ratings = ratings.copy()
    ratings["movie_string"] = ratings.apply(lambda x: f"movie_{x['movie_id']}", axis=1)
    ratings["rating_string"] = ratings.apply(lambda x: f"rating_{x['rating']}", axis=1)

    sequences = ratings.groupby("user_id").apply(
        lambda x: ["<BOS>"] + [item for pair in zip(x['movie_string'], x['rating_string']) for item in pair],
        include_groups=False
    )
    return sequences

def train_test_split(sequences, test_ratio=TEST_DATA_RATIO):
    n_ratings = sum([(len(seq) -1) / 2 for seq in sequences])
    n_ratings_to_test = int(n_ratings * test_ratio)
    test_ratings_per_user = n_ratings_to_test // len(sequences)

    print("Final test ratio", test_ratings_per_user * len(sequences) / n_ratings)
    print("Number of test ratings per user", test_ratings_per_user)

    # For train take all movies and ratings except the last one
    train_sequences = sequences.apply(lambda x: x[:-test_ratings_per_user*2])
    # For test take all movies and ratings
    test_sequences = sequences
    return train_sequences, test_sequences, test_ratings_per_user

# Tokenizer class for encoding and decoding movie and rating information
class Tokenizer:
    def __init__(self):
        # Initialize the vocabulary with the bos_token and ratings tokens
        self.stoi = {
            "rating_1": 0,
            "rating_2": 1,
            "rating_3": 2,
            "rating_4": 3,
            "rating_5": 4,
            "<BOS>": 5, # Beginning of sequence
            "<UNK>": 6,
        }

    def train(self, movie_ids):

        unique_movie_ids = set(movie_ids)
        self.stoi.update({f"movie_{movie_id}": idx for idx, movie_id in enumerate(unique_movie_ids, start=len(self.stoi))})
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        print("Vocabulary size:", self.vocab_size)

    def encode(self, sequence):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in sequence]

    def decode(self, indices):
        return [self.itos.get(idx, "<UNK>") for idx in indices]

def get_continuous_sequence(sequences):
    continuous_sequence = [item for seq in sequences for item in seq]
    return continuous_sequence

def get_random_batch(continuous_sequence, batch_size=32, block_size=256):
    sequence_length = len(continuous_sequence)
    x_list, y_list = [], []

    for _ in range(batch_size):
        start_index = torch.randint(0, sequence_length - block_size - 1, (1,)).item()
        # Ensure the sequence ends on a rating token
        while continuous_sequence[start_index + block_size] >= 6 or continuous_sequence[start_index + block_size] == 0:
            start_index = start_index - 1

        end_index = start_index + block_size
        x_list.append(continuous_sequence[start_index:end_index])
        y_list.append(continuous_sequence[start_index + 1:end_index + 1])

    return torch.tensor(x_list, dtype=torch.long), torch.tensor(y_list, dtype=torch.long)

def pad_sequence(sequences: np.array, batch_first=True, padding_value=0, padding_position="left"):
    """Returns a torch.tensor of shape (batch_size, max_length) where the sequences are padded to the <padding_position> with the padding value."""
    # Find the length of the longest sequence
    max_length = max([len(seq) for seq in sequences])

    # Create a new tensor with the same number of dimensions as the input sequences
    output = torch.full((len(sequences), max_length), padding_value)

    # Fill the output tensor with the sequences
    for i, seq in enumerate(sequences):
        if padding_position == "left":
            output[i, -len(seq):] = torch.tensor(seq)
        else:
            output[i, :len(seq)] = torch.tensor(seq)

    if batch_first:
        return output
    else:
        return output.T

def get_batch_padded(train_sequences, batch_size=32, how="random"):
    sorted_train_sequences = sorted(train_sequences, key=len)
    
    # Create batches
    batches = [sorted_train_sequences[i:i + batch_size] for i in range(0, len(sorted_train_sequences), batch_size)]
    n_batches = len(batches)
    # Pad sequences within each batch
    padded_batches = []
    for batch in batches:
        # Pad sequences on the left with zeros
        padded_batch = pad_sequence(batch, padding_value=5)
        padded_batches.append(padded_batch)

    batch_index = 0
    while True:
        if how == "random":
            random_batch_index = torch.randint(0, len(padded_batches), (1,)).item()
            batch = padded_batches[random_batch_index]
        if how == "sequential":
            batch = padded_batches[batch_index]
            batch_index = (batch_index + 1) % n_batches
        # Create x and y
        x = batch[:, :-1]
        y = batch[:, 1:]

        yield x, y

def evaluate(model, test_tokenized_sequences, n_samples=100, device="cpu"):
    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for sequence in np.random.choice(test_tokenized_sequences, n_samples, replace=False):
            x = torch.tensor(sequence[:-1], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(sequence[1:], dtype=torch.long).unsqueeze(0).to(device)
            output, _ = model(x)
            
            loss = F.cross_entropy(output[:, :, -1:], y[:, -1:])
            total_loss += loss.item()
        
    return total_loss / n_samples

def evaluate_rmse(model, batch_generator, n_samples=100, device="cpu", test_ratings_per_sequence=1):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_n_sequences = 0
        for _ in range(n_samples):
            x, y = next(batch_generator)
            output, _ = model(x.to(device))
            ratings_output = torch.argmax(output[:, -test_ratings_per_sequence*2:, :5], dim=-1)
            ratings_target = y[:, -test_ratings_per_sequence*2:].to(device)

            # Take every other item, which are the ratings
            ratings_output = ratings_output[:, 1::2]
            ratings_target = ratings_target[:, 1::2]

            loss = F.mse_loss(ratings_output.float(), ratings_target.float())
            # Each batch can be of variable size so we need to weight the loss
            total_loss += loss.item() * x.size(0)
            total_n_sequences += x.size(0)
    model.train()

    return np.sqrt(total_loss / total_n_sequences)


def main():
    # Load and preprocess data
    users, ratings, movies = load_data()
    print("Number of ratings", len(ratings))

    sequences = ratings_to_sequences(ratings)
    train_sequences, test_sequences, test_ratings_per_user = train_test_split(sequences)

    # Initialize tokenizer and prepare sequences
    tokenizer = Tokenizer()
    tokenizer.train(movies["movie_id"].unique())

    # Process sequences
    train_tokenized_sequences = train_sequences.apply(tokenizer.encode).values
    test_tokenized_sequences = test_sequences.apply(tokenizer.encode).values
    

    continuous_train_sequence = get_continuous_sequence(train_tokenized_sequences)
    # Generate a random batch for training
    x, y = get_random_batch(continuous_train_sequence)
    random_batch = next(get_batch_padded(train_tokenized_sequences))
    print(random_batch[0].shape, random_batch[1].shape)
    print(random_batch)

    # Initialize the model
    model = RandomRatingGenerator(tokenizer.vocab_size)
    eval_loss = evaluate(model, test_tokenized_sequences)
    print(eval_loss)
    rmse = evaluate_rmse(model, test_tokenized_sequences, 100, "cpu", test_ratings_per_user), 
    print("RMSE: ", rmse)


if __name__ == "__main__":
    main()