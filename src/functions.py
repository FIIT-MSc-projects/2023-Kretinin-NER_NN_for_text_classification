import os
import pickle

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from math import nan

from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Attention
import keras
import tensorflow as tf
# print(tf.__version__)
# print(tfa.__version__)
# from keras_contrib.layers import CRF
from keras import Model, Input
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import SpatialDropout1D
import nltk
from nltk.stem import WordNetLemmatizer

from keras import regularizers
from time import sleep
from tqdm import tqdm


from bs4 import BeautifulSoup
import requests
from zipfile import ZipFile
import re
from datetime import date


def split_text(text):
    sentences = nltk.sent_tokenize(text)
    return [nltk.word_tokenize(sentence) for sentence in sentences]


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmatized_tokens


def lemmatize_sentences(sentence):
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in nltk.word_tokenize(sentence)]

    return ' '.join(lemmatized_tokens)


def preprocess_dataset(dataset_name, word_column, sentence_column, tag_column, other_tag):
    # data = pd.read_csv(dataset_name, encoding="latin1")
    # data = data.fillna(method="ffill")

    data = pd.read_parquet('kaggle/input/cord-ner-full.parquet.gzip')

    # words = list(set(data["Word"].values))
    words = list(set(data[word_column].values))
    words.append("ENDPAD")
    num_words = len(words)

    tags = list(set(data[tag_column].values))
    num_tags = len(tags)

    class SentenceGetter(object):
        def __init__(self, data):
            self.n_sent = 1
            self.data = data
            self.empty = False
            agg_func = lambda s: [(w, t) for w, t in zip(s[word_column].values.tolist(), s[tag_column].values.tolist())]
            self.grouped = self.data.groupby(sentence_column).apply(agg_func)
            self.sentences = [s for s in self.grouped]

        def get_next(self):
            try:
                s = self.grouped["{}".format(self.n_sent)]
                self.n_sent += 1
                return s
            except:
                return None

    getter = SentenceGetter(data)
    sentences = getter.sentences

    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    max_len = data.groupby([sentence_column], sort=False).size().max()

    x = [[word2idx[w[0]] for w in s] for s in sentences]
    x = pad_sequences(maxlen=max_len, sequences=x, padding="post", value=word2idx["ENDPAD"])

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx[other_tag])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # y_train = to_categorical(y_train, num_tags)
    # y_test = to_categorical(y_test, num_tags)

    return x_train, x_test, y_train, y_test, word2idx, tag2idx, max_len, words, tags


def build_matrix_embeddings(path, num_tokens, embedding_dim, word_index):


    hits, misses = 0, 0
    embeddings_index = {}

    print('Loading file...')

    sleep(0.5)

    for line in tqdm(open(path, encoding='utf-8')):
        word, coefs = line.split(maxsplit=1)
        embeddings_index[word] = np.fromstring(coefs, "f", sep=" ")

    print("Processed %s Word Vectors." % len(embeddings_index))

    sleep(0.5)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    for word, i in tqdm(word_index.items()):
        if i >= num_tokens:
            continue
        try:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                embedding_vector = embeddings_index.get(str(word).lower())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    embedding_vector = embeddings_index.get(str(word).upper())
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                        hits += 1
                misses += 1
        except:
            embedding_matrix[i] = embeddings_index.get('UNK')

    print("Hits: %d Tokens | Miss: %d Tokens" % (hits, misses))

    return embedding_matrix


def ner_2(x_train, x_test, y_train, y_test, word2idx, tag2idx, max_len, words, tags, words_weights):
    print('Tensorflow version:', tf.__version__)
    print('GPU detected:', tf.config.list_physical_devices('GPU'))

    idx2tag = {v: k for k, v in tag2idx.items()}
    idx2word = {i: w for w, i in word2idx.items()}

    input_word = Input(shape=(max_len,))

    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")

    with strategy.scope():

        model = Embedding(input_dim=len(words), output_dim=100, input_length=max_len)(input_word)
        # model = Embedding(input_dim=words_weights.shape[0], output_dim=words_weights.shape[1],
        #                   weights=[words_weights], trainable=True)(input_word)
        # model = SpatialDropout1D(0.2)(model)
        rnn_1 = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0))(model)
        rnn_drop = Dropout(0.1)(rnn_1)
        rnn_2 = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0))(rnn_drop)
        # attention = Attention()([rnn_2, rnn_2])
        # hidden1 = keras.layers.Dense(units=len(tags), activation='relu')(rnn_2)
        dense_1 = TimeDistributed(Dense(len(tags) * 4, activation="relu"))(rnn_2)
        dense_2 = TimeDistributed(Dense(len(tags) * 2, activation="relu"))(dense_1)
        out = keras.layers.Dense(units=len(tags), activation='softmax')(dense_2)
        model = Model(input_word, out)
        model.summary()

        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(
            x=x_train,
            y=y_train,
            validation_split=0.1,
            batch_size=128,
            epochs=10,
            verbose=1
        )

    model.evaluate(x_test, y_test)

    model.save('models/my_model4/model.h5')

    with open('models/my_model4/history.pickle', 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('models/my_model4/tags.pickle', 'wb') as handle:
        pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('models/my_model4/words.pickle', 'wb') as handle:
        pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(3):
        #i = np.random.randint(0, x_test.shape[0])  # 659
        p = model.predict(np.array([x_test[i]]))
        p = np.argmax(p, axis=-1)
        y_true = y_test[i]
        print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
        print("-" * 30)
        for w, true, pred in zip(x_test[i], y_true, p[0]):
            if words[w] == "ENDPAD":
                break
            print("{:15}{}\t{}".format(words[w], tags[true], tags[pred]))
        print("----------------------------")

    visualize_history(history)

    return


def visualize_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()


def plot_history(history):
    plt.style.use('ggplot')
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, 'b', label='Training acc')
    plt.plot(x, val_accuracy, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# The same one as in the test_environment.py
def progress_bar(iteration, total):
    total_len = 100
    percent_part = ("{0:.2f}").format(100 * (iteration / total))
    filled = int(total_len * iteration / total)
    bar = '█' * filled + '-' * (total_len - filled)
    print(f'\r Progress: [{bar}] {percent_part}%', end='')
    if iteration == total:
        print()


#  https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558
def get_last_N_articles(number):
    print("If progress bar don't move for a long time, then 'class' parameter for the BeatuifulSoup find_all() should be revised and actualized")
    # current_page = "https://www.reuters.com/news/archive/worldNews?view=page&page=1&pageSize=10"
    current_page = "https://www.theguardian.com/lifeandstyle/health-and-wellbeing?page=1"

    processed_num = 0

    page_counter = 1

    # Main page ref
    main_page = "https://www.theguardian.com/"
    page_first_part = "https://www.theguardian.com/lifeandstyle/health-and-wellbeing?page="
    page_second_part = "&pageSize=10"

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    zipArch = ZipFile('data/articles_' + str(date.today()) + "_" + str(number) + '.zip', 'w')
    tmp_file_path = "data/"

    while (processed_num < number):

        r1 = requests.get(current_page)
        coverpage = r1.content

        soup1 = BeautifulSoup(coverpage, "html5lib")
        regex = re.compile('u-faux-block-link__overlay')
        coverpage_news = soup1.find_all('a', {"class": regex})

        page_counter += 1

        # current_page = page_first_part + str(page_counter) + page_second_part
        current_page = page_first_part + str(page_counter)


        # each page contains 20 articles
        for iter in np.arange(0, 20):
            progress_bar(processed_num, number)
            if (processed_num == number):
                break

            # Getting the link of the article
            link = coverpage_news[iter].attrs['href']
            list_links.append(link)

            # Getting the title
            title = coverpage_news[iter].get_text()
            list_titles.append(title)

            # Reading the content (it is divided in paragraphs)
            article = requests.get(link)
            article_content = article.content
            soup_article = BeautifulSoup(article_content, 'html5lib')
            regex = re.compile('article-body-.*')
            body = soup_article.find_all('div', {"class": regex})

            # Skip if page doesn't have "article" part
            if len(body) == 0:
                continue
            x = body[0].find_all('p')

            # Unifying the paragraphs
            list_paragraphs = []
            final_article = ""

            # Store text of article, if article contents it
            if len(x) > 0:
                processed_num += 1
                for p in np.arange(0, len(x)):
                    paragraph = x[p].get_text()
                    list_paragraphs.append(paragraph)
                final_article = " ".join(list_paragraphs)

                news_contents.append(final_article)

                # Open file(create), write text inside it, move its copy to the zip, and delete this file
                tmp_file = open(tmp_file_path + "Article" + str(processed_num) + ".txt", "w", encoding="utf-8")
                tmp_file.write(final_article)
                tmp_file.close()
                zipArch.write(tmp_file.name, os.path.basename(tmp_file.name))
                os.remove(tmp_file_path + "Article" + str(processed_num) + ".txt")

    # close the Zip File
    zipArch.close()
