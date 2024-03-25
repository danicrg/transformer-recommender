from data import load_data, ratings_to_sequences, train_test_split, Tokenizer, get_continuous_sequence, get_random_batch, evaluate_rmse, get_batch_padded
import math

import torch
from model import GPTModel, RandomRatingGenerator, AverageRatingGenerator

from tqdm import tqdm

torch.manual_seed(1337)
torch.mps.empty_cache()

device = "mps" if torch.backends.mps.is_available() else "cpu"

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

# Training
users, ratings, movies = load_data()
sequences = ratings_to_sequences(ratings)
train_sequences, test_sequences, test_ratings_per_user = train_test_split(sequences)

# Initialize tokenizer and prepare sequences
tokenizer = Tokenizer()
tokenizer.train(movies["movie_id"].unique())

class Config:
    vocab_size = tokenizer.vocab_size
    n_steps = 1000
    batch_size = 2
    n_embed = 128
    block_size = max([len(seq) for seq in train_sequences])
    dropout = 0.2
    n_layer = 2
    n_head = 2
    learning_rate = 4e-4
    criteria =  "cross_entropy" # One of ["cross_entropy", "masked_cross_entropy", "rating_masked_cross_entropy", "distance_weighted_loss"]
    device = device

config = Config()

# Process sequences
train_tokenized_sequences = train_sequences.apply(tokenizer.encode).values
test_tokenized_sequences = test_sequences.apply(tokenizer.encode).values


train_batch_generator = get_batch_padded(train_tokenized_sequences, batch_size=config.batch_size, how="random")
test_batch_generator = get_batch_padded(test_tokenized_sequences, batch_size=config.batch_size//2, how="sequential")
n_test_batches = len(test_tokenized_sequences) // (config.batch_size//2) if len(test_tokenized_sequences) % (config.batch_size//2) == 0 else len(test_tokenized_sequences) // (config.batch_size//2) + 1

continuous_train_sequence = get_continuous_sequence(train_tokenized_sequences)

random_model = RandomRatingGenerator(tokenizer.vocab_size)
average_model = AverageRatingGenerator(tokenizer.vocab_size)
random_model_eval_loss = evaluate_rmse(random_model, test_batch_generator, n_test_batches, "cpu", test_ratings_per_user)
average_model_eval_loss = evaluate_rmse(average_model, test_batch_generator, n_test_batches, "cpu", test_ratings_per_user)
print("Random model RMSE: ", random_model_eval_loss)
print("Average model RMSE: ", average_model_eval_loss)
print("Number of train_sequences: ", len(train_sequences))


m = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)

print("Number of non-overlapping sequences in the training set: ", len(continuous_train_sequence)/config.block_size)
print("Steps per epoch:", len(train_sequences)/config.batch_size)
print("Number of parameters: ", sum(p.numel() for p in m.parameters() if p.requires_grad))

n_epochs = 3
n_steps = n_epochs * len(train_sequences) // config.batch_size

warmup_iters = n_steps // 300
lr_decay_iters = n_steps
min_lr = config.learning_rate / 10
decay_lr = True

def train():    
    min_train_loss = float("inf")
    last_step = 0
    min_eval_loss = float("inf")
    for i in tqdm(range(n_steps)):
        
        lr = get_lr(i) if decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # xb, yb = get_random_batch(continuous_train_sequence)
        xb, yb = next(train_batch_generator)
        _, loss = m(xb.to(device), yb.to(device))

        if loss.item() < min_train_loss and i > last_step + 100 and i < n_steps - 1:
            last_step = i
            min_train_loss = loss.item()
            n_batches_to_test = n_test_batches
            eval_loss = evaluate_rmse(m, test_batch_generator, n_batches_to_test, device, test_ratings_per_user)
            min_eval_loss = eval_loss # Not the actual min eval loss but the corresponding train loss
            min_train_loss = min(min_train_loss, loss.item())
            print(f"Train Loss: {loss.item()} Test RMSE: {eval_loss}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    eval_loss = evaluate_rmse(m, test_batch_generator, n_test_batches, device, test_ratings_per_user)
    
    print("Loss function: ", config.criteria)
    print("Final Test RMSE: ", eval_loss)
    print("Min Train Loss: ", min_train_loss)
    print("Min Test RMSE: ", min_eval_loss)

    return eval_loss, min_train_loss, min_eval_loss

def select_loss():
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

train()