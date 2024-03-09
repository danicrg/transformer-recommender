from data import load_data, preprocess_data, train_test_split, Tokenizer, get_continuous_sequence, get_random_batch, evaluate_rmse
import math

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

continuous_train_sequence = get_continuous_sequence(train_tokenized_sequences)

class Config:
    vocab_size = tokenizer.vocab_size
    n_steps = 1000
    batch_size = 64
    n_embed = 64
    block_size = 512
    dropout = 0.1
    n_layer = 1
    n_head = 1
    learning_rate = 1e-3
    criteria =  "cross_entropy" # One of ["cross_entropy", "masked_cross_entropy", "rating_masked_cross_entropy", "distance_weighted_loss"]
    device = device


config = Config()


m = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)

print("Number of non-overlapping sequences in the training set: ", len(continuous_train_sequence)/config.block_size)
print("Number of parameters: ", sum(p.numel() for p in m.parameters() if p.requires_grad))

n_steps = max(config.n_steps, 2 * len(continuous_train_sequence)//config.block_size)

warmup_iters = n_steps // 300
lr_decay_iters = n_steps
min_lr = config.learning_rate / 10
decay_lr = True

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return config.learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (config.learning_rate - min_lr)

def train():
    min_train_loss = float("inf")
    min_eval_loss = float("inf")
    for i in tqdm(range(n_steps)):
        
        lr = get_lr(i) if decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        xb, yb = get_random_batch(continuous_train_sequence, config.batch_size, config.block_size)
        _, loss = m(xb.to(device), yb.to(device))

        if i % 100 == 0:
            n_samples = len(test_tokenized_sequences) if i % 1000 == 0 else 100
            eval_loss = evaluate_rmse(m, test_tokenized_sequences, n_samples, device)
            min_eval_loss = min(min_eval_loss, eval_loss)
            min_train_loss = min(min_train_loss, loss.item())
            print(f"Train Loss: {loss.item()} Test RMSE: {eval_loss}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    eval_loss = evaluate_rmse(m, test_tokenized_sequences, len(test_tokenized_sequences), device)
    
    print("Loss function: ", config.criteria)
    print("Final Test RMSE: ", eval_loss)
    print("Min Train Loss: ", min_train_loss)
    print("Min Test RMSE: ", min_eval_loss)

    return eval_loss, min_train_loss, min_eval_loss

results = {}

for criteria in ["cross_entropy", "masked_cross_entropy", "rating_masked_cross_entropy", "distance_weighted_loss"]:
    config.criteria = criteria
    m = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)
    print(f"Training with {criteria} loss")
    eval_loss, min_train_loss, min_eval_loss = train()
    print("\n\n\n")

    results[criteria] = {
        "eval_loss": eval_loss,
        "min_train_loss": min_train_loss,
        "min_eval_loss": min_eval_loss
    }

    print(results)
