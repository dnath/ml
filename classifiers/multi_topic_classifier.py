import logging

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report

import numpy as np

from nltk import word_tokenize
from nltk.stem import LancasterStemmer, PorterStemmer

class StemmingTokenizer:
    def __init__(self, stemmer_type='Porter'):
        self.stemmer_type = stemmer_type
        if self.stemmer_type == 'Porter':
            self.stemmer = PorterStemmer()
        elif self.stemmer_type == 'Lancaster':
            self.stemmer = LancasterStemmer()
        else:
            raise Exception('Invalid stemmer_type = {0}'.format(stemmer_type))

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]

class MultiTopicClassifier:
    def __init__(self, labeled_corpus, topics, training_fraction=1):
        self.labeled_corpus = labeled_corpus
        self.topics = topics
        self.training_fraction = training_fraction
        self.vectorizer = self.get_vectorizer()
        self.topic_models = {}

    def get_vectorizer(self):
        return TfidfVectorizer(min_df=1,
                               ngram_range=(1, 2),
                               stop_words='english',
                               # tokenizer=StemmingTokenizer(),
                               strip_accents='unicode',
                               norm='l2'
                               # use_idf=True,
                               # smooth_idf=True,
                               # sublinear_tf=True
                               # max_features=40000
                              )

    def build_models(self):
        trainset_size = int(round(len(labeled_corpus) * self.training_fraction))

        X_train = np.array([''.join(text) for topic_list, text in labeled_corpus[0:trainset_size]])
        X_test = np.array([''.join(text) for topic_list, text in labeled_corpus[trainset_size+1:len(labeled_corpus)]])

        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)

        for topic in topics:
            logging.debug('\n=======================================[{0}]===========================================\n\n'.format(topic))

            y_train = np.array([1 if topic in topic_list else 0 for topic_list, text in self.labeled_corpus[0:trainset_size]])

            y_train_1 = 0
            y_train_0 = 0
            for y in y_train:
                if y == 1:
                    y_train_1 += 1
                else:
                    y_train_0 += 1


            y_test = np.array([1 if topic in topic_list else 0 for topic_list, text in self.labeled_corpus[trainset_size+1:len(labeled_corpus)]])

            logging.debug('y_train_0 =', y_train_0)
            logging.debug('y_train_1 =', y_train_1)

            over_sample_weight = y_train_0 / y_train_1
            logging.debug('f =', over_sample_weight)

            self.topic_models[topic] = MultinomialNB().fit(X_train,
                                                           y_train,
                                                           sample_weight=np.array([over_sample_weight if label == 1 else 1 for label in y_train]))

            if self.training_fraction != 1:
                self.test_model(topic, X_test, y_test)


    def test_model(self, topic, X_test, y_test):
        y_predicted = self.topic_models[topic].predict(X_test)
        logging.debug('The precision for this classifier is {0}'.format(metrics.precision_score(y_test, y_predicted)))
        logging.debug('The recall for this classifier is {0}'.format(metrics.recall_score(y_test, y_predicted)))
        logging.debug('The f1 for this classifier is {0}'.format(metrics.f1_score(y_test, y_predicted)))
        logging.debug('The accuracy for this classifier is {0}'.format(metrics.accuracy_score(y_test, y_predicted)))

        logging.debug('\nHere is the classification report: \n{0}'.format(classification_report(y_test, y_predicted)))
        logging.debug('\nHere is the confusion matrix:\n{0}'.format(metrics.confusion_matrix(y_test,
                                                                                             y_predicted,
                                                                                             labels=[0, 1])))

    def get_top_n_topics(self, unlabeled_data, n=10):
        top_n_topics = []

        for data in unlabeled_data:
            X_eval = self.vectorizer.transform(np.array([data]))

            scores = {topic: self.topic_models[topic].predict_proba(X_eval)[0][1] for topic in self.topics}
            scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)

            top_n = [k for k, _ in scores[:n]]
            top_n_topics.append(top_n)

        return top_n_topics



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    line = raw_input()
    line_segs = line.split()

    T = int(line_segs[0]) # number of labeled data items
    E = int(line_segs[1]) # number of unlabeled data items to evaluate

    labeled_corpus = [(raw_input().split()[1:], raw_input()) for t in xrange(T)]
                        # of the form [([label1, label2, ...], text), ... ]

    label_list = [t for (t, _) in labeled_corpus]
    topics = list(set().union(*label_list))

    mtc = MultiTopicClassifier(labeled_corpus, topics, training_fraction=1)
    mtc.build_models()

    data_to_eval = [raw_input() for e in xrange(E)]

    for top_n_topics in mtc.get_top_n_topics(data_to_eval):
        print top_n_topics