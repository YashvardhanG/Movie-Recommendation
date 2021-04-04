import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

def index(index):
        return df[df.index == index]["title"].values[0]

def title(title):
        return df[df.title == title]["index"].values[0]

df = pd.read_csv("Dataset.csv")
features = ['keywords','cast','genres','director']

for feature in features:
        df[feature] = df[feature].fillna('')

def combine_features(row):
        try:
                return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
        except:
                print("Error:", row)	

df["combined_features"] = df.apply(combine_features,axis=1)

cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix)

count = len(df["original_title"])
movie = random.randint(0,count)
dict = {}
dict = df["original_title"]
user = (dict[movie])

print("\nMovie user likes:",user)

chosen_movie = user

print("\nTop 10 Recommended movies to the user:")
movie_index = title(chosen_movie)

similar_movies =  list(enumerate(cosine_sim[movie_index]))

movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
for element in movies:
        print(index(element[0]))
        i=i+1
        if i>10:
                break
        else:
                pass
