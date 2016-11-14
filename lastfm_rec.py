from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

df = pd.read_csv('artists_500.csv')

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['bio'].values.astype('U'))
feature_names = np.array(tfidf.get_feature_names())

def top_tfidf(row_num, n):
	row = tfidf_matrix[row_num]
	tfidf_sorting = np.argsort(row.toarray()).flatten()[::-1]
	top_n = feature_names[tfidf_sorting][:n]
	return top_n

# print(top_tfidf(70, 20))

def cos_sim(artist_id):
	cosine_similarity = linear_kernel(tfidf_matrix[artist_id], tfidf_matrix).flatten()
	top_sim = cosine_similarity.argsort()[:-10:-1]
	s = [df.ix[top_sim[i]]['artist'] for i in range(len(top_sim))]
	return s

print(df.ix[34]['artist'] + ' is similar to:')
print(cos_sim(34))

