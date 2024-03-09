# transformer-recommender
A recommender based on language modeling with a GPT like model


## Results
### Experiment 1
Fixed block size. 

Loss: custom_distance_weighted_loss_with_mask

0.58M parameters

    batch_size = 64
    n_embed = 64
    block_size = 512
    dropout = 0.1
    n_layer = 1
    n_head = 1
    learning_rate = 3e-4
    mask_movies = True

n_steps = 3895

Train Loss: 0.11226 
Final Test RMSE: 1.1789
Min Test RMSE: 1.0

### Experiment 2
Same as Experiment 1

    learning_rate = 1e-3


Train Loss: 0.1134 
Final Test RMSE: 1.2020
Min Test RMSE: 1.06

### Experiment 3
Same as 2

LR shedule

Train Loss: 0.11349
Final Test RMSE: 1.192484
Min Test RMSE: 0.979795