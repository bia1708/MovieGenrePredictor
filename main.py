import pandas as pd
from sklearn.tree import DecisionTreeClassifier

raw_data = pd.read_csv("titles.csv")
raw_data = raw_data.dropna().reset_index(drop=True)


data = raw_data.drop(columns=['id', 'title', 'release_year', 'description', 'age_certification', 'runtime', 'seasons', 'type', 'production_countries', 'imdb_id', 'imdb_votes', 'tmdb_popularity', 'tmdb_score'])

# data: title, genres, imdb_score

input_data = data.drop(columns='genres')
output_data = data['genres']

model = DecisionTreeClassifier()
model.fit(input_data.values, output_data)

while True:
    try:
        user_input = float(input("Enter a score: "))
        if user_input > 10 or user_input < 0:
            raise Exception

        prediction = model.predict([[user_input]])
        output = prediction[0].split(" ")

        print(f"Genres most likely to receive a score of {user_input}: ")
        for word in output:
            print('\t' + word.strip("[''],"))
        print('\n')

    except ValueError:
        print("Please enter a valid number!\n")
    except Exception:
        print("Please enter a valid score!\n")
