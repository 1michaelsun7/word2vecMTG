import collections
import json
import math
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
import string
import sys
import tensorflow

def build_dataset(words, vocabulary_size = 50000):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data_index = 0

def generate_batch(data, batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window  # target label at the center of the buffer
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels

def main():
	color_dict = {"W": 0, "U": 1, "B": 2, "R": 3, "G": 4, "C": 5}

	with open('MTGcardtextcolors.json') as json_file:
		json_data = json.load(json_file)
		json_file.close()

	sentences = []
	targets = []
	for key in json_data.keys():
		card = json_data[key]
		cardtext = card["text"].encode('utf-8')
		for punct in string.punctuation:
			cardtext = cardtext.replace(punct, "")
		standardized_text = cardtext.lower()

		targets.append(card["colors"])
		sentences.append(standardized_text.split())

	val_cutoff = int(math.floor(0.9*len(sentences)))

	train_set = (sentences[:val_cutoff], targets[:val_cutoff])
	val_set = (sentences[val_cutoff:], targets[val_cutoff:])

	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
								stop_words = None, max_features = 50000) 
	train_data_features = vectorizer.fit_transform(train_set[0])
	train_data_features = train_data_features.toarray()
	print train_data_features.shape

	#SVM_model 

	# word_model = gensim.models.Word2Vec(sentences, min_count=5)
	# print word_model.most_similar(positive=['deal', 'damage'], negative=['cards'])

if __name__ == '__main__':
	main()