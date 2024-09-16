import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Input, Dot, Concatenate, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
books = pd.read_csv("data/Books.csv")
ratings = pd.read_csv("data/Ratings.csv")
users = pd.read_csv("data/Users.csv")

# Rename columns for books dataframe
books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher'}, inplace=True)

# Rename columns for users dataframe
users.rename(columns={'User-ID': 'user_id', 'Location': 'location', 'Age': 'age'}, inplace=True)

# Rename columns for ratings dataframe
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)

# Filter ratings based on the number of ratings per user
user_ratings_count = ratings['user_id'].value_counts()
selected_users = user_ratings_count[user_ratings_count > 200].index
filtered_ratings = ratings[ratings['user_id'].isin(selected_users)]
rating_with_books = filtered_ratings.merge(books, on='ISBN')

# Sample data for faster execution (you can increase the sample size)
df = rating_with_books.sample(n=10000, random_state=42)

# Data Preprocessing
user_encoder = LabelEncoder()
isbn_encoder = LabelEncoder()

df['user_id'] = user_encoder.fit_transform(df['user_id'])
df['ISBN'] = isbn_encoder.fit_transform(df['ISBN'])

# Split data into training and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Collaborative Filtering Model
embedding_size = 50

user_input = Input(shape=(1,))
isbn_input = Input(shape=(1,))

user_embedding = Embedding(len(user_encoder.classes_), embedding_size)(user_input)
isbn_embedding = Embedding(len(isbn_encoder.classes_), embedding_size)(isbn_input)

user_flat = Flatten()(user_embedding)
isbn_flat = Flatten()(isbn_embedding)

dot_product = Dot(axes=1)([user_flat, isbn_flat])

model = Model(inputs=[user_input, isbn_input], outputs=dot_product)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([train_data['user_id'], train_data['ISBN']], train_data['rating'], epochs=10, batch_size=64, verbose=1)

# Content-Based Filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books['title'].values.astype('U'))

# Find the ISBN of the input book titles
user_input_title = 'Finding Moon'
input_isbn = df[df['title'] == user_input_title]['ISBN'].values
if len(input_isbn) == 0:
    print("Book title not found in the dataset.")
    exit()

# Calculate cosine similarity between the input book and all books
cosine_sim = linear_kernel(tfidf_matrix, tfidf_vectorizer.transform([user_input_title]))
cosine_scores = list(enumerate(cosine_sim[0]))

# Sort the books based on similarity scores
cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)

# Get top N similar books
top_n = 5
similar_books = cosine_scores[1:top_n + 1]

# Generate book recommendations for the user
user_id_to_recommend_for = 10  # Replace with the user ID you want to recommend books to
user_isbns = df[df['user_id'] == user_id_to_recommend_for]['ISBN'].unique()

# Generate collaborative filtering recommendations
isbn_ids = np.arange(len(isbn_encoder.classes_))
user_ids = np.full(len(isbn_encoder.classes_), user_id_to_recommend_for)
ratings = model.predict([user_ids, isbn_ids]).flatten()

# Sort collaborative filtering recommendations by rating (descending order)
collaborative_recommendations = pd.DataFrame({'ISBN': isbn_ids, 'rating': ratings})
collaborative_recommendations = collaborative_recommendations[~collaborative_recommendations['ISBN'].isin(user_isbns)]
collaborative_recommendations = collaborative_recommendations.sort_values(by='rating', ascending=False)

# Get top N collaborative filtering recommended books
top_n_collaborative = collaborative_recommendations.head(5)

# Get top N content-based recommendations
top_n_content_based = [books.iloc[i[0]]['ISBN'] for i in similar_books]

# Combine collaborative and content-based recommendations
combined_recommendations = list(set(top_n_collaborative['ISBN']).union(set(top_n_content_based)))

# Print the top combined recommendations
combined_recommendation_results = []
for isbn in combined_recommendations:
    book_info = books[books['ISBN'] == isbn]
    title = book_info['title'].values[0]
    image_link = book_info['Image-URL-S'].values[0]
    combined_recommendation_results.append({'title': title, 'image_url': image_link})

print(combined_recommendation_results)
