import pandas as pd
from sklearn.neighbors import NearestNeighbors

books = pd.read_csv("data/Books.csv")
ratings = pd.read_csv("data/Ratings.csv")

# Rename columns for books dataframe
books = books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher'})



# Rename columns for ratings dataframe
ratings = ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'})


# Filter ratings based on the number of ratings per user
user_ratings_count = ratings['user_id'].value_counts()
selected_users = user_ratings_count[user_ratings_count > 200].index
filtered_ratings = ratings[ratings['user_id'].isin(selected_users)]
rating_with_books = filtered_ratings.merge(books, on='ISBN')
rating_with_books

df = rating_with_books.head(10000)

def get_top_recommendations():
    # Recommendation logic (similar to the previous code)
    # Filter out books with low ratings (e.g., only consider books with ratings >= 3)
    df_filtered = df[df['rating'] >= 3]

    # Group by book title and calculate the average rating and count
    book_stats = df_filtered.groupby('title')['rating'].agg(['mean', 'count'])

    # Rename the columns for clarity
    book_stats.rename(columns={'mean': 'average_rating', 'count': 'rating_count'}, inplace=True)

    # Merge the book_stats with the original DataFrame
    df_filtered = df_filtered.merge(book_stats, left_on='title', right_index=True)

    # Define a function to calculate a weighted ranking score
    def weighted_ranking(row):
        return (row['average_rating'] * row['rating_count'])

    # Calculate the weighted ranking score for each book
    df_filtered['ranking'] = df_filtered.apply(weighted_ranking, axis=1)

    # Sort the DataFrame by the ranking score in descending order
    df_filtered = df_filtered.sort_values(by='ranking', ascending=False)

    # Get the top 5 recommended books
    top_recommended_books = df_filtered.drop_duplicates(subset='title').head(5)

    # Extract book titles and image URLs
    recommended_titles = top_recommended_books['title']
    recommended_image_urls = top_recommended_books['Image-URL-S']

    # Return the recommendations
    return recommended_titles.tolist(), recommended_image_urls.tolist()