import os
import re 
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string 
import numpy as np
stemmer = SnowballStemmer('english')
vectorizer = CountVectorizer(analyzer="word")
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
forest = RandomForestClassifier(n_estimators=100)

data = []
labels = []

train_data = []
train_labels = []

test_data = []
test_labels = []

def categoryToLabel(category):
	if category == 'business':
		return 1
	elif category == 'entertainment':
		return 2
	elif category == 'politics':
		return 3
	elif category == 'sport':
		return 4
	elif category == 'tech':
		return 5

def labelToCategory(label):
	if label == 1:
		return 'business'
	elif label == 2:
		return 'entertainment'
	elif label == 3:
		return 'politics'
	elif label == 4:
		return 'sport'
	elif label == 5:
		return 'tech'



def threeDigits(num):
	three_digits_value = "%03d"%num
	return str(three_digits_value)

def cleanUpData(category,num):
	path = 'bbc/' + category + '/' + num + '.txt'
	data_file = open(path)
	raw_text = data_file.read()

	# remove stuff which isnt characters on numbers
	parsed = re.sub("[^a-zA-Z]", " ", raw_text)
	parsed_list = parsed.split()

	# stem the words
	stemmed = []
	for word in parsed_list:
		stemmed_word = stemmer.stem(word)
		stemmed.append(stemmed_word)

	# remove stopwords and keep meaningful words
	stop_words = set(stopwords.words('english'))
	meaningful_words = [w for w in stemmed if not w in stop_words]
	meaningful_string = " ".join(meaningful_words)

	# add data and labels to 'data' and 'labels' variables 
	data.append(meaningful_string)
	label = categoryToLabel(category)
	labels.append(label)

	# close the file
	data_file.close()

def createData(start,end):
	for i in xrange(start,end):
		for category in categories:
			num = threeDigits(i)
			cleanUpData(category, num)

def vectorize(data):
	vectorized_data = vectorizer.fit_transform(data)
	vectorized_data = vectorized_data.toarray()
	# if you want to see the vocabulary
	#vocab = vectorizer.get_feature_names()
	return vectorized_data

# print the words and the number of times they appear
#dist = np.sum(train_data_features,axis=0)
#for tag, count in zip(dist, vocab):
#	print tag, count
print 'about to create data'
createData(1,380)
print 'finished creating data'
data = vectorize(data)

def sliceData(data,labels,num_train):
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []

	length = len(data)
	for i in xrange(0,length):
		if i < num_train:
			train_data.append(data[i])
			train_labels.append(labels[i])
		elif i >= num_train:
			test_data.append(data[i])
			test_labels.append(labels[i])
	return {'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data, 'test_labels': test_labels}


#print 'labels: ', labels

all_data = sliceData(data,labels,1700)
print 'all_data:', all_data
train_data = all_data['train_data']
train_labels = all_data['train_labels']
test_data = all_data['test_data']
test_labels = all_data['test_labels']


print 'length of train_data: ', len(train_data)
print 'length of train_labels: ', len(train_labels)

print 'length of test_data: ', len(test_data)
print 'length of test_labels: ', len(test_labels)


forest = forest.fit(train_data,train_labels)

score = 0

for n in xrange(0,100):
	prediction = forest.predict(test_data[n])
	prediction = prediction[0]
	correct_answer = test_labels[n]
	correct_answer = int(correct_answer)
	if int(prediction) == int(correct_answer):
		score += 1

print 'score: ', score


#score = forest.score(test_data, test_labels)
#print 'score: ', score








