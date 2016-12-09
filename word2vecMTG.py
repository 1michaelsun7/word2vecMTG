import collections
import json
import math
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import string
import sys
#import tensorflow

print string.punctuation

def cross_entropy_loss(predicted, true):
    return -1.0*np.sum(np.multiply(true, np.log(predicted + 1e-16)))

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
    # modified_punct = string.punctuation.replace("+", "")
    # modified_punct = modified_punct.replace("-", "")
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
        sentences.append(standardized_text)
        #sentences.append(standardized_text.split())

    val_cutoff = int(math.floor(0.9*len(sentences)))

    train_set = (sentences[:val_cutoff], targets[:val_cutoff])
    val_set = (sentences[val_cutoff:], targets[val_cutoff:])

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    print "==========Initializing bag-of-words...=========="
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                stop_words = None, max_features = 5000) 
    train_features = vectorizer.fit_transform(train_set[0])
    train_features = train_features.toarray()
    train_labels = np.array(train_set[1], dtype="|S6")

    val_features = vectorizer.transform(val_set[0])
    val_features = val_features.toarray()
    val_labels = np.array(val_set[1])
    print train_features.shape
    print train_labels.shape

    vocab = vectorizer.get_feature_names()
    dist = np.sum(train_features, axis=0)

    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    # for tag, count in zip(vocab, dist):
    #     print count, tag


    # Random Forest Classifier
    #
    #
    #
#    print "==========Initializing random forest...=========="
#    forest = RandomForestClassifier(n_estimators = 100) 
#
#    forest = forest.fit(train_features, train_labels)
#
#    result = forest.predict(val_features)
#    correct = np.sum(np.multiply(result, val_labels), axis=1)
#    print len(correct[correct == 1])
#    print np.sum(correct)
#    CE = cross_entropy_loss(val_labels, result)
#    print "Cross Entropy Loss: ", CE
    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(train_features, train_labels)


if __name__ == '__main__':
    main()