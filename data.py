from zipfile import ZipFile
from urllib.request import urlretrieve
import pandas as pd


# urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
# ZipFile("movielens.zip", "r").extractall()

# Ideas:
# Mask the movies when calculating the loss vs not masking


users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    encoding="ISO-8859-1",
    engine="python",
)

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
    encoding="ISO-8859-1",
    engine="python",
)

movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    names=["movie_id", "title", "genres"],
    encoding="ISO-8859-1",
    engine="python",
)

print("# of movies: ", len(movies))

print(ratings)
print(len(users))

ratings.sort_values(by=["user_id", "unix_timestamp"], inplace=True)
lengths = ratings.groupby("user_id").apply(lambda x: len(x))
print(lengths.describe())

sequences = ratings.groupby("user_id").apply(
    lambda x: " ".join([f"movie_{mid} rating_{r}" for mid, r in zip(x.movie_id, x.rating)])
)

print(sequences.iloc[1])
