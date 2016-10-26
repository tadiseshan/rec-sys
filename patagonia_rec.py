import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df = pd.read_csv('patagonia_data.csv')


def train(df, id):
	tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
	tfidf_matrix = tf.fit_transform(df['description'])
	cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
	related_docs_indices = cosine_similarities[id].argsort()[:-10:-1]
	
	return related_docs_indices

print(train(df, 0))