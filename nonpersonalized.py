import pandas as pd

#set up dataframes
ratings = pd.read_csv('data/ratings.csv')
tags = pd.read_csv('data/tags.csv', encoding = 'ISO-8859-1')
movies = pd.read_csv('data/movies.csv', encoding='ISO-8859-1')

#create merged dataframes and groupby object, grouping ratings by title
df = pd.merge(movies, ratings, on='movieId')
groupby_rating = df['rating'].groupby(df['title'])

#calculate rating means for all items and return highest 20
def itemMeans(gb):
	return gb.mean().sort_values(ascending=False).head(20)

#calculate damped means for individual movie ids
def dampedMeans(movieId, k):
	movie = df['movieId'] == movieId
	sumRatings = df[movie]['rating'].sum()
	globalMean = df.rating.mean()
	numRatings = len(df[movie]['rating'])
	return ((sumRatings)+(globalMean * k))/(numRatings + k)

#add a function to recommend top damped means? or damped means for each movieId

#basic association model (P(X intersection Y)/P(Y))
def basicAssoc(i_id,j_id):
	j = df['movieId'] == j_id
	i = df['movieId'] == i_id
	j_users = pd.Series(df[j]['userId'].unique())
	i_users = pd.Series(df[i]['userId'].unique())
	total_users = len(df['userId'].unique())

	#calculate i and j intersection
	inter = set(i_users).intersection(set(j_users))
	return (len(inter)/total_users)/((len(j_users)/total_users))

#lift association model (P(X intersection Y)/(P(Y)*P(X)))
def liftAssoc(i_id,j_id):
	j = df['movieId'] == j_id
	i = df['movieId'] == i_id
	j_users = pd.Series(df[j]['userId'].unique())
	i_users = pd.Series(df[i]['userId'].unique())
	total_users = len(df['userId'].unique())

	#calculate i and j intersection
	inter = set(i_users).intersection(set(j_users))
	return (len(inter)/total_users)/((len(j_users)/total_users)*(len(i_users)/total_users))

#to do: write functions that take input and provide recommendations

