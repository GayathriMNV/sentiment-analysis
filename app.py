import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template
import pickle
import nltkmodules

# Initializing Flask
app = Flask(__name__)

# Load reviews clean data,tfidf, LogisticRegression and User Based Recommendation pickle files
clean_data = pickle.load(open('reviews_clean_data.pkl', 'rb'))
tfidf_vect_model = pickle.load(open('tfidf.pkl', 'rb'))
logistic_model = pickle.load(open('LogisticRegression_Model.pkl', 'rb'))
user_recommendation_model = pickle.load(open('user_based_sentiment_model.pkl', 'rb'))


@app.route('/')

def home():
	return render_template('index.html')

@app.route("/predict", methods = ['POST'])

def predict():
	if( request.method == 'POST'):

		user_name = request.form['reviews_username']
		user_name = user_name.lower()

		# If user_name doesnot exist in the dataset
		if user_name not in user_recommendation_model.index :
			return render_template('index.html', prediction_text = 'Enter a valid User Name')

		# if user_name exists in the dataset
		user_top20_products = user_recommendation_model.loc[user_name].sort_values(ascending = False)[:20]
		user_top20_products = pd.DataFrame(user_top20_products)
		user_top20_products.reset_index(inplace = True)

		# Merge the products and reviews  based on the sentiment
		top20_products_recommended = pd.merge(user_top20_products, clean_data, on = ['name'])

		# Featrue extraction using tfidf model
		tfidf_sentiment = tfidf_vect_model.transform(top20_products_recommended['processed_reviews'])

		# Prediction with Logistic regression Model
		lr_prediction = logistic_model.predict(tfidf_sentiment)

		# Top 20 predicted Products
		top20_products_recommended['top20_products_pred'] = lr_prediction

		# Find the Positive sentiment percentage
		sentiment_score = top20_products_recommended.groupby(['name'])['top20_products_pred'].agg(['sum','count']).reset_index()
		sentiment_score['percent'] = round((100*sentiment_score['sum'] / sentiment_score['count']),2)
		
		# Predicting Top 5 Products
		# Top 5 products based on sentiment score.
		sentiment_score = sentiment_score.sort_values(by='percent',ascending=False)
		top5_products_recommended = sentiment_score['name'].head().to_list()
		
		
		return render_template('index.html', user_name = user_name, prediction_text = '1.{} 2.{} 3.{} 4.{}5.{}'.format(top5_products_recommended[0],
																									top5_products_recommended[1],
																									top5_products_recommended[2],
																									top5_products_recommended[3],
																									top5_products_recommended[4],
																									)
							)
		
		
	else :
		return render_template('index.html')	


if __name__ == "__main__":
	app.run(debug = True)
