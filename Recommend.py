import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def standardize(row) :
     new_row = (row - row.mean()) / (row.max() - row.min())
     return new_row


def build_input(song_name, user_rating) :
     #input matrix user X items
     A = np.random.randint(5, size=(100,100))
     data = pd.DataFrame(A)
     data.columns=["User"+str(i) for i in range(1, 101)]
     data.index=["Song"+str(i) for i in range(1, 101)]

     # standardized dataframe
     ratings_std = data.apply(standardize)

     item_similarity = cosine_similarity(ratings_std)
     item_sim_df = pd.DataFrame(item_similarity, index = data.index, columns = data.index)
     print(item_sim_df)

     similar_score = item_sim_df[song_name] * (user_rating - 2.5)
     similar_score = similar_score.sort_values(ascending = False)
     return similar_score


if __name__ == '__main__':
    print(build_input("Song86",5))
