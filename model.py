
'''
# Sentiment-Based Product Recommendation system, which includes the following tasks,

1.Data sourcing and sentiment analysis
2.Building a recommendation system
3.Improving the recommendations using the sentiment analysis model
4.Deploying the end-to-end project with a user interface

'''
#  Importing packages,Reading & Understanding Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
import pickle as pickle

# Importing libraries for text preprocessing and analysis
import re, nltk, spacy, string
from nltk.corpus import stopwords
#nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Importing libraries for Models Building
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import f1_score, classification_report,precision_score,recall_score,confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score

# Importing libraries to balance Class
from collections import Counter
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE

# Remove warnings
import warnings
warnings.filterwarnings('ignore')


# Display options - row/column display limit
#pd.set_option('max_columns', None)
#pd.set_option('max_rows', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)

# 1.Data sourcing and sentiment analysis

# Loading data 'sample30.csv' to ebuzz_data dataframe
ebuzz_data = pd.read_csv('sample30.csv')

'''
# 1.1 Data cleaning
	1.1.1 Missing values Handling/ Data Qulaity check the columns
	1.1.2 Reviews Text data Cleaning
'''
# 1.1.1 Missing values Handling/ Data Qulaity check the columns

# Lets check the missing values percentages in each column/variable
round((ebuzz_data.isnull().sum()/len(ebuzz_data.index)*100),2) 

# Target variable user_sentiment has 1 missing value, lets impute the unknown sentiment as'0' 

# Lets map Positive as '1' and Negative sentiments as '0'
ebuzz_data['user_sentiment'] = ebuzz_data['user_sentiment'].apply(lambda x : 1 if x == "Positive" else 0)

# Typecasting from float to int datatype
ebuzz_data['user_sentiment'] = ebuzz_data['user_sentiment'].astype('int64')

# Lets replace the reviews_title with 'Null' values
ebuzz_data['reviews_title'].fillna('' , inplace = True)

# Lets create a new colum 'reviews' by merging the reviews_text and reviews_title columns
ebuzz_data['reviews'] =  ebuzz_data['reviews_title'] +" "+ ebuzz_data['reviews_text']

# Lets drop the existing reviews_text and reviews_title columns
ebuzz_data.drop(['reviews_text' , 'reviews_title'] , axis = 1, inplace = True)

# Let's create a 'ebuzz_clean_data' dataframe
ebuzz_clean_data = ebuzz_data[['name','reviews_username','reviews_rating','user_sentiment','reviews']]

# Since the missing data for reviews_username is less than 63, we can drop the records
ebuzz_clean_data.dropna(inplace = True)


'''
1.2. Text preprocessing
Stopwords removal
puctuation removal
stemming/lemmatization
'''
# 1.2 Text preprocessing

# Creating a function to clean the text and remove all the unnecessary elements.

def clean_text(text):
    
    #Make the text lowercase
    text_lower = text.lower()
     
    # To remove 1), 2) like paterns
    text_no_curve = re.sub(r'^\d[1]\)', '',text_lower) 
        
    # HTML Tags removal
    text_no_html = re.sub(r'<.*?>' , '', text_no_curve)
        
    #Remove punctuation & special character
    text_nospl = re.sub(r'[?|!|\'|"|#|.|,|)|(|\|/|~|%|*-]', '', text_no_html)
    
    # Remove words containing numbers
    #text_no_nums = re.sub(r'\w*\d\w*', '', text_nospl) 
    
    return text_nospl

# Create a new feature "processed_reviews" by cleaning the 'reviews' column
ebuzz_clean_data['processed_reviews'] = ebuzz_clean_data['reviews'].apply(lambda x: clean_text(x))

# Lets a write a function to lemmatize the reviews text
def lemmatize_text(text):
    stopwords_list = set(stopwords.words('english'))
    lm = WordNetLemmatizer()
    tokenised_text = nltk.word_tokenize(text)
    lemmatised_text = [lm.lemmatize(word, pos = 'v') for word in tokenised_text if word not in stopwords_list]
    clean_text = ' '.join(lemmatised_text)
    return clean_text

# Lets apply the lemmatization on the processed_reviews
ebuzz_clean_data['processed_reviews'] = ebuzz_clean_data['processed_reviews'].apply(lambda x : lemmatize_text(x)) 

# Lets convert the clean data file to a cvs file
ebuzz_clean_data.to_csv('ebuzz_clean_data.csv')

# Lets save the data in pickle file for future use
pickle.dump(ebuzz_clean_data, open("reviews_clean_data.pkl","wb"))


'''
# 1.3. Training a sentiment Analysis model:
	1.3.1 Feature extraction using tf-idf Vectorizer
	1.3.2 Handling Class Imbalance using SMOTE
	1.3.3 Building & Training Sentiment Model

 '''

 # 1.3.1 Feature extraction using tf-idf Vectorizer
X = ebuzz_clean_data['processed_reviews']
y = ebuzz_clean_data['user_sentiment']

# Splitting the data into train & test
X_train, X_test, y_train, y_test = train_test_split( X,y ,test_size = 0.3, random_state = 45, stratify = y)


# Applying Tfidtransformer
tfidf_model= TfidfVectorizer(max_features=3000, lowercase=True, analyzer='word', stop_words= 'english', ngram_range =(1,2) )
tf_X_train = tfidf_model.fit_transform(X_train).toarray()
tf_X_test = tfidf_model.transform(X_test)

# Dump the tfidf vectorizer to pkl file
pickle.dump(tfidf_model, open("tfidf.pkl", "wb"))


# 1.3.2 Handling Class Imbalance using SMOTE

# Applying  SMOTE 

counter = Counter(y_train)

#oversampling using SMOTE
smote = SMOTE(random_state = 45)
X_train_sm, y_train_sm = smote.fit_resample(tf_X_train, y_train)

counter = Counter(y_train_sm)


# 1.3.3 Building & Training Sentiment Model - Logistic Regression Model

# Lets create a logistic Regression Model

# Create an Logistic Regresion object
lr_model = LogisticRegression()

params_grid = {'C':[10, 1, 0.5, 0.1],
               'penalty':['l1','l2'],
               'class_weight':['balanced']
              }


# Create grid search using 4-fold cross validation
lr_grid_search = GridSearchCV( estimator = lr_model, 
                            param_grid = params_grid, 
                            cv=4, 
                            scoring='roc_auc',
                            n_jobs=-1)

# Lets tune the model with best gridsearch estimators
lr_tuned = lr_grid_search.fit(X_train_sm, y_train_sm)

# Logitic model evalution
y_prob_test_lr = lr_tuned.predict_proba(tf_X_test)
y_pred_test_lr = lr_tuned.predict(tf_X_test)

#print('Test Score:')
#print('Confusion Matrix')
#print('\n')
confusion = confusion_matrix(y_test,y_pred_test_lr)
#print(confusion,"\n")
TP = confusion[1,1]  # true positive
TN = confusion[0,0]  # true negatives
FP = confusion[0,1]  # false positives
FN = confusion[1,0]  # false negatives
sensitivity= TP / float(TP+FN)
specificity = TN / float(TN+FP)
accuracy = accuracy_score(y_test,y_pred_test_lr)
precision = precision_score(y_test,y_pred_test_lr)
f1 = f1_score(y_test,y_pred_test_lr)
#print("Sensitivity : ",sensitivity )
#print("Specificity : ",specificity )
#print('Classification Report')
#print('-'*60)
#print(classification_report(y_test,y_pred_test_lr),"\n")
#print('AUC-ROC=',roc_auc_score(y_test, y_prob_test_lr[:,1]))
    
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_prob_test_lr[:,1])

AUC_ROC_LR = roc_auc_score(y_test, y_prob_test_lr[:,1])

# Lets dump the tuned_logisticregression model to a pickle file
pickle.dump(lr_tuned, open("LogisticRegression_Model.pkl", "wb"))


# 2. Building a recommendation system
'''
	2.1 Building a User based recommendation
	2.2 User based prediction 
	2.3 User based Evaluation
'''
# 2.1 Building a User based recommendation

# lets load the raw product reviews data
reviews_data = pd.read_csv('sample30.csv')

# Lets copy the ebuzz_data for the User based recommendation model
ubr_data = reviews_data[['reviews_username','name','reviews_rating']]

ubr_data = ubr_data[~ubr_data['reviews_username'].isna()]

# Let's split the data into Train & Test of the dataset.
train, test = train_test_split(ubr_data, test_size=0.30, random_state=31)


# Pivot the train ubr_data' dataset into matrix format in which columns are name and reviews_rating and the rows are reviews_username.
ubr_pivot = train.pivot_table( index='reviews_username',
                               columns='name',
                               values='reviews_rating'
                             ).fillna(0)


# Creating dummy train & dummy test dataset
# Copy the train dataset into dummy_train
dummy_train = train.copy()

# The movies not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)

# Convert the dummy train dataset into matrix format.
dummy_train_ubr = pd.pivot_table( data = dummy_train, 
                                 index='reviews_username',
                                 columns='name',
                                 values='reviews_rating' 
                                ).fillna(1)


# User Similarity Matrix - adjusted Cosine

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(ubr_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0


# Create a user-product matrix.
ubr_pivot = train.pivot_table( index='reviews_username',
                                     columns='name',
                                     values='reviews_rating' 
                             )

# Normalising the rating of the product for each user around 0 mean
mean = np.nanmean(ubr_pivot, axis=1)
ubr_subtracted = (ubr_pivot.T-mean).T

# Find cosine similarity: Used pairwise distance to find similarity.

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(ubr_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0

# 2.2 User Based Prediction

# Ignore the correlation for values less than 0.
user_correlation[user_correlation < 0] = 0

# Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the review rating 
# (as present in the rating dataset).

user_predicted_ratings = np.dot(user_correlation, ubr_pivot.fillna(0))

# user_final_rating this contains predicted ratings for products
user_final_rating = np.multiply(user_predicted_ratings,dummy_train_ubr)

### Finding the top 5 recommendation for the user

# Take the user ID as input
#user_input = input("Enter your user name : ")
user_input = 'aaron'

# Recommended products for the selected user based on ratings
user_recommendation = user_final_rating.loc[user_input].sort_values(ascending=False)[:20]

# Lets load the raw data and merge it with the user_input related product recommendation
data = pd.read_csv('sample30.csv')

# Merging the user_recommendation and the data files on 'name'
user_recommendation = pd.merge(user_recommendation ,data, left_on='name',right_on='name', how = 'left')

# 2.3 User Based Evaluation 

# Find out the common users of test and train dataset.
common_users = test[test.reviews_username.isin(train.reviews_username)]

# convert into the user-product matrix.
common_ubr_matrix = common_users.pivot_table(index='reviews_username', 
                                            columns='name', 
                                            values='reviews_rating')

# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)

user_correlation_df['userId'] = ubr_subtracted.index
user_correlation_df.set_index('userId',inplace=True)

list_name = common_users.reviews_username.tolist()
user_correlation_df.columns = ubr_subtracted.index.tolist()
user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]

user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

user_correlation_df_3 = user_correlation_df_2.T

user_correlation_df_3[user_correlation_df_3 < 0] = 0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_ubr_matrix.fillna(0))

dummy_test_ubr = common_users.copy()

dummy_test_ubr['reviews_rating'] = dummy_test_ubr['reviews_rating'].apply(lambda x: 1 if x >= 1 else 0)

dummy_test_ubr = dummy_test_ubr.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)

common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test_ubr)


# RMSE
# Calculating RMSE only for the products rated by user. For RMSE, normalising the rating to (1,5) range.


X  = common_user_predicted_ratings.copy() 
X = X[X > 0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

common_ = common_users.pivot_table(index='reviews_username', columns='name', values='reviews_rating')

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse_ubr = round((sum(sum((common_ - y )**2))/total_non_nan)**0.5, 2)


# User Based Recommendation system is the best recommendation model with very low RMSE.

# Lets convert the User Recommendation system results into a CSV file
user_final_rating.to_csv('user_based_sentiment.csv')

# Lets dump the results of the user_based_recommendation to a pickle file
pickle.dump(user_final_rating, open('user_based_sentiment_model.pkl','wb'))


# 3.Recommendation of Top 20 Products to a Specified User

# load all pickle files
tfidf_vect_model = pickle.load(open('tfidf.pkl', 'rb'))
user_recommendation_model = pickle.load(open('user_based_sentiment_model.pkl', 'rb'))
logistic_model = pickle.load(open('LogisticRegression_Model.pkl', 'rb'))

# Enter user name
# user = input("Enter User Name : ")
user = 'aaron'

# Recommend top 20 products
user_top20_products = user_recommendation_model.loc[user].sort_values(ascending=False)[:20]

user_top20_products = pd.DataFrame(user_top20_products)
user_top20_products.reset_index(inplace = True)
#user_top20_products

# merge top 20 products and its reviews
top20_products_sentiment = pd.merge(user_top20_products,ebuzz_clean_data, on = ['name'])
#top20_products_sentiment.head()

# convert text to feature
top20_products_tfidf = tfidf_vect_model.transform(top20_products_sentiment['processed_reviews'])

# model prediction
top20_products_recommended = logistic_model.predict(top20_products_tfidf)
top20_products_recommended

top20_products_sentiment['top20_products_pred'] = top20_products_recommended

sentiment_score = top20_products_sentiment.groupby(['name'])['top20_products_pred'].agg(['sum','count']).reset_index()
sentiment_score['percent'] = round((100*sentiment_score['sum'] / sentiment_score['count']),2)
#sentiment_score.head(20)


# Top 5 products:

# Top 5 products based on sentiment score.
sentiment_score = sentiment_score.sort_values(by='percent',ascending=False)
top5_products_recommended = sentiment_score['name'].head().to_list()
