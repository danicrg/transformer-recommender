from data import load_data, preprocess_data, train_test_split, Tokenizer, get_continuous_sequence, get_random_batch, evaluate_rmse

import torch
from model import GPTModel, RandomRatingGenerator, AverageRatingGenerator

from tqdm import tqdm

torch.manual_seed(1337)
torch.mps.empty_cache()

device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"

# Training
users, ratings, movies = load_data()
sequences = preprocess_data(ratings)
train_sequences, test_sequences = train_test_split(sequences)

# Initialize tokenizer and prepare sequences
tokenizer = Tokenizer()
tokenizer.train(movies["movie_id"].unique())

# Process sequences
train_tokenized_sequences = train_sequences.apply(tokenizer.encode).values
test_tokenized_sequences = test_sequences.apply(tokenizer.encode).values

random_model = RandomRatingGenerator(tokenizer.vocab_size)
average_model = AverageRatingGenerator(tokenizer.vocab_size)
random_model_eval_loss = evaluate_rmse(random_model, test_tokenized_sequences, len(test_tokenized_sequences), "cpu")
average_model_eval_loss = evaluate_rmse(average_model, test_tokenized_sequences, len(test_tokenized_sequences), "cpu")
print("Random model RMSE: ", random_model_eval_loss)
print("Average model RMSE: ", average_model_eval_loss)

continuous_train_sequence = get_continuous_sequence(train_sequences, tokenizer)

class Config:
    vocab_size = tokenizer.vocab_size
    n_steps = 1000
    batch_size = 64
    n_embed = 128
    block_size = 512
    dropout = 0.1
    n_layer = 4
    n_head = 4
    learning_rate = 4e-4
    mask_movies = True
    device = device


config = Config()

m = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)

print("Number of non-overlapping sequences in the training set: ", len(continuous_train_sequence)/config.block_size)
print("Number of parameters: ", sum(p.numel() for p in m.parameters() if p.requires_grad))

n_steps = max(config.n_steps, len(continuous_train_sequence)//config.block_size)

for i in tqdm(range(n_steps)):

    xb, yb = get_random_batch(continuous_train_sequence, config.batch_size, config.block_size)
    _, loss = m(xb.to(device), yb.to(device))

    if i % 20 == 0:
        n_samples = len(test_tokenized_sequences) if i % 100 == 0 else 100
        eval_loss = evaluate_rmse(m, test_tokenized_sequences, n_samples, device)
        print(f"Train Loss: {loss.item()} Test RMSE: {eval_loss}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

m.eval()
