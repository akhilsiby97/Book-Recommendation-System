import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Input, Concatenate, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_recommendations(title,user_id):
    books = pd.read_csv("data/Books.csv")
    ratings = pd.read_csv("data/Ratings.csv")

    books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
    ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)


    x = ratings['user_id'].value_counts() > 200
    y = x[x].index  #user_ids
    ratings = ratings[ratings['user_id'].isin(y)]
    rating_with_books = ratings.merge(books, on='ISBN')
    df = rating_with_books.head(10000)
    # Data Preprocessing
    user_encoder = LabelEncoder()
    isbn_encoder = LabelEncoder()

    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['ISBN'] = isbn_encoder.fit_transform(df['ISBN'])

    # Split data into training and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Filter the test data to include only user IDs and ratings present in the training data
    train_user_ids = train_data['user_id'].unique()
    test_data_filtered = test_data[test_data['user_id'].isin(train_user_ids)]
    test_data_filtered = test_data_filtered[test_data_filtered['rating'].isin(train_data['rating'])]

    # Create the neural network model
    embedding_size = 50

    user_input = Input(shape=(1,))
    isbn_input = Input(shape=(1,))

    user_embedding = Embedding(len(user_encoder.classes_), embedding_size)(user_input)
    isbn_embedding = Embedding(len(isbn_encoder.classes_), embedding_size)(isbn_input)

    user_flat = Flatten()(user_embedding)
    isbn_flat = Flatten()(isbn_embedding)

    dot_product = tf.keras.layers.Dot(axes=1)([user_flat, isbn_flat])

    model = Model(inputs=[user_input, isbn_input], outputs=dot_product)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit([train_data['user_id'], train_data['ISBN']], train_data['rating'], epochs=10, batch_size=64, verbose=1)

    # Find the ISBN of the input book title
    user_input_title = title
    input_isbn = df[df['title'] == user_input_title]['ISBN'].values
    if len(input_isbn) == 0:
        print("Book title not found in the dataset.")
        return []

    # Generate book recommendations for the user based on the input book
    user_id_to_recommend_for = user_id  # Replace with the user ID you want to recommend books to
    user_isbns = df[df['user_id'] == user_id_to_recommend_for]['ISBN'].unique()

    # Generate book recommendations for the user
    isbn_ids = np.arange(len(isbn_encoder.classes_))
    user_ids = np.full(len(isbn_encoder.classes_), user_id_to_recommend_for)
    ratings = model.predict([user_ids, isbn_ids]).flatten()

    # Sort book recommendations by rating (descending order)
    recommendations = pd.DataFrame({'ISBN': isbn_ids, 'rating': ratings})
    recommendations = recommendations[~recommendations['ISBN'].isin(user_isbns)]  # Exclude books already rated by the user
    recommendations = recommendations.sort_values(by='rating', ascending=False)

    # Get top N recommended books (titles and image links)
    top_n_recommendations = recommendations.head(5)

    # Print the top recommended books (titles and image links)
    recommendation_results = []
    for _, row in top_n_recommendations.iterrows():
        isbn = row['ISBN']
        book_info = df[df['ISBN'] == isbn]
        title = book_info['title'].values[0]
        image_link = book_info['Image-URL-S'].values[0]
        recommendation_results.append({'title': title, 'image_url': image_link})
    return recommendation_results