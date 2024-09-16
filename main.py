from google.cloud import datastore
import datetime
import random
from flask import Flask, render_template, redirect
import google.oauth2.id_token
from flask import Flask, render_template, request
from google.auth.transport import requests
from recommendation import get_top_recommendations
from tensorflowmodel import get_recommendations
app = Flask(__name__)

datastore_client = datastore.Client()
firebase_request_adapter = requests.Request()
    
@app.route('/')
def root():
  recommended_titles, recommended_image_urls = get_top_recommendations()

    # Prepare the data to pass to the template
  recommendations = [{'title': title, 'image_url': image_url} for title, image_url in zip(recommended_titles, recommended_image_urls)]
  
  return render_template('index.html',recommendations = recommendations)

@app.route('/book_search',methods=['POST'])
def movie_search():
 #movie_title = "Harry Potter and the Sorcerer's Stone (Book 1)"
 title = request.form['Search_book']
 user_id = random.randint(100, 200)
 recommendations = get_recommendations(title,user_id)
 return render_template('index.html', recommendations = recommendations)

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8080, debug=True)
