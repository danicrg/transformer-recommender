from zipfile import ZipFile
from urllib.request import urlretrieve
import os

# Third-party imports
import pandas as pd
import torch

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

def preprocess_data(ratings):
    # Sort ratings by user_id and unix_timestamp
    ratings.sort_values(by=["user_id", "unix_timestamp"], inplace=True)
    # Reserve a portion of ratings for testing
    train = ratings[:-int(TEST_DATA_RATIO * len(ratings))]
    test = ratings[-int(TEST_DATA_RATIO * len(ratings)):]
    return train, test

# Tokenizer class for encoding and decoding movie and rating information
class Tokenizer:
    def __init__(self):
        # Initialize the vocabulary with the bos_token and ratings tokens
        self.stoi = {
            "<BOS>": 0,
            "rating_1": 1,
            "rating_2": 2,
            "rating_3": 3,
            "rating_4": 4,
            "rating_5": 5,
            "<UNK>": 6,
        }

    def train(self, movie_ids):
        unique_movie_ids = set(movie_ids)
        self.stoi.update({f"movie_{id}": idx for idx, id in enumerate(unique_movie_ids, start=len(self.stoi))})
        self.itos = {idx: token for token, idx in self.stoi.items()}
        print("Vocabulary size:", len(self.stoi))

    def encode(self, sequence):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in sequence]

    def decode(self, indices):
        return [self.itos.get(idx, "<UNK>") for idx in indices]


def prepare_sequences(ratings, tokenizer):
    ratings["movie_string"] = ratings.apply(lambda x: f"movie_{x['movie_id']}", axis=1)
    ratings["rating_string"] = ratings.apply(lambda x: f"rating_{x['rating']}", axis=1)

    tokenizer.train(ratings["movie_string"].unique())

    sequences = ratings.groupby("user_id").apply(
        lambda x: ["<BOS>"] + [item for pair in zip(x['movie_string'], x['rating_string']) for item in pair]
    )

    tokenized_sequences = sequences.apply(tokenizer.encode).tolist()
    continuous_sequence = [item for seq in tokenized_sequences for item in seq]
    return continuous_sequence

def get_random_batch(continuous_sequence, batch_size=32, block_size=128):
    sequence_length = len(continuous_sequence)
    x_list, y_list = [], []

    for _ in range(batch_size):
        start_index = torch.randint(0, sequence_length - block_size - 1, (1,)).item()
        # Ensure the sequence ends on a relevant note
        end_index = start_index + block_size
        x_list.append(continuous_sequence[start_index:end_index])
        y_list.append(continuous_sequence[start_index + 1:end_index + 1])

    return torch.tensor(x_list, dtype=torch.long), torch.tensor(y_list, dtype=torch.long)

def main():
    # Load and preprocess data
    users, ratings, movies = load_data()
    train_ratings, test_ratings = preprocess_data(ratings)

    # Initialize tokenizer and prepare sequences
    tokenizer = Tokenizer()
    continuous_train_sequence = prepare_sequences(train_ratings, tokenizer)
    continuous_test_sequence = prepare_sequences(test_ratings, tokenizer)

    # Generate a random batch for training
    x, y = get_random_batch(continuous_train_sequence)
    print(x, y)

if __name__ == "__main__":
    main()