# This is a sample Python script.
import pickle

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
import sys
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import zipfile

import functions


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


text = """Avul Pakir Jainulabdeen Abdul Kalam was born on 15 October 1931, to a Tamil Muslim family in the pilgrimage centre of Rameswaram on Pamban Island, then in the Madras Presidency and now in the State of Tamil Nadu. His father Jainulabdeen Marakayar was a boat owner and imam of a local mosque;[9] his mother Ashiamma was a housewife.[10][11][12][13] His father owned a ferry that took Hindu pilgrims back and forth between Rameswaram and the now uninhabited Dhanushkodi.[14][15] Kalam was the youngest of four brothers and one sister in his family.[16][17][18] His ancestors had been wealthy Marakayar traders and landowners, with numerous properties and large tracts of land. Even though his ancestors had been wealthy Marakayar traders, the family had lost most of its fortunes by the 1920s and was poverty-stricken by the time Kalam was born. Marakayar are a Muslim ethnic found in coastal Tamil Nadu and Sri Lanka who claim descendance from Arab traders and local women. Their business had involved trading groceries between the mainland and the island and to and from Sri Lanka, as well as ferrying pilgrims between the mainland and Pamban. As a young boy he had to sell newspapers in order to add to the family's meager income. With the opening of the Pamban Bridge to the mainland in 1914, however, the businesses failed and the family fortune and properties were lost over time, apart from the ancestral home."""

text3 = """The history of timekeeping devices dates back to when ancient civilizations first observed astronomical bodies as they moved across the sky. Devices and methods for keeping time have since then improved through a long series of new inventions and ideas. Sundials and water clocks originated from ancient Egypt, and were later used by the Babylonians, the Greeks and the Chinese. In the medieval period, Islamic water clocks were unrivalled in their sophistication until the mid-14th century. Incense clocks were being used in China by the 6th century. The hourglass, one of the few reliable methods of measuring time at sea, was a European invention and does not seem to have been used in China before the mid-16th century. In medieval Europe, purely mechanical clocks were developed after the invention of the bell-striking alarm, used to warn a man to toll the monastic bell. The weight-driven mechanical clock, controlled by the action of a verge and foliot, was a synthesis of earlier ideas derived from European and Islamic science, and one of the most important inventions in the history of timekeeping. The most famous mechanical clock was designed and built by Henry de Vick in c. 1360—for the next 300 years, all the improvements in timekeeping were essentially developments based on it. The invention of the mainspring in the early 15th century allowed small clocks to be built for the first time. From the 17th century, the discovery that clocks could be controlled by harmonic oscillators led to the most productive era in the history of timekeeping. Leonardo da Vinci had produced the earliest known drawings of a pendulum in 1493–1494, and in 1582 Galileo Galilei had investigated the regular swing of the pendulum, discovering that frequency was only dependent on length. The pendulum clock, designed and built by Dutch polymath Christiaan Huygens in 1656, was so much more accurate than other kinds of mechanical timekeepers that few clocks have survived with their verge and foliot mechanisms intact. Other innovations in timekeeping during this period include inventions for striking clocks, the repeating clock and the deadbeat escapement. Errors in early pendulum clocks were eclipsed by those caused by temperature variation, a problem tackled during the 18th century by the English clockmakers John Harrison and George Graham. Only the invention of invar in 1895 eliminated the need for such innovations."""


def preprocess_text(text, word2idx, max_len):
    # Split the text into tokens
    sentences = functions.split_text(text)

    # lemmatizer = WordNetLemmatizer()
    #
    # def get_wordnet_pos(treebank_tag):
    #     if treebank_tag.startswith('J'):
    #         return wordnet.ADJ
    #     elif treebank_tag.startswith('V'):
    #         return wordnet.VERB
    #     elif treebank_tag.startswith('N'):
    #         return wordnet.NOUN
    #     elif treebank_tag.startswith('R'):
    #         return wordnet.ADV
    #     else:
    #         return None

    # Convert the tokens to integer IDs using the word2id dictionary
    ids = []
    endpad_idx = word2idx['ENDPAD']
    for tokens in sentences:
        array = []
        for token in tokens:
            if token in word2idx.keys():
                array.append(word2idx[token])
            else:
                array.append(0)

        # array = [word2idx[lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag))] if lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) in word2idx else 0 for token, tag in tokens_with_tags]
        while len(array) < max_len:
            array.append(endpad_idx)
        ids.append(array)

    return ids, sentences


def print_result(text, result, idx2tag):
    tags = np.argmax(result, axis=-1)
    for sentence in text:
        print("----------------")
        print("{:15}\t {}\n".format("Word", "Pred"))
        print("-" * 30)
        for w, pred in zip(sentence, tags[0]):
            print("{:15}\t{}".format(w, idx2tag[pred]))


def plot_history_pickle(filename):
    with open(filename, 'rb') as handle:
        history = pickle.load(handle)
        functions.plot_history(history)


def predict_text(processed, model, tags):
    for i in range(3):
        p = model.predict(np.array([processed[0][i]]))
        p = np.argmax(p, axis=-1)
        print(p)
        print("{:15}\t {}\n".format("Word", "Pred"))
        print("-" * 30)
        for idx, pred in enumerate(p[0][0:len(processed[1][i])]):
            print("{:15}\t {}".format(processed[1][i][idx], tags[pred]))
        print("----------------------------")


def predict_text_2(text, model, tags):
    result = []
    for i in range(len(text)):
        p = model.predict(np.array([text[0][i]]), verbose=0)
        p = np.argmax(p, axis=-1)
        for idx, pred in enumerate(p[0][0:len(text[1][i])]):
            if tags[pred] != "Other":
                result.append((text[1][i][idx], tags[pred]))
    return result


def process_file(file, filename, texts):
    content = file.read(filename)
    if type(content) == bytes:
        text = content.decode('utf-8')
        texts.append(text)

    if len(content.strip()) == 0:
        print("No text was found")
        return


def get_labeled_words(model, texts, word2idx):
    fin_result = []
    for text in texts:
        processed = preprocess_text(text, word2idx, model.layers[0].output_shape[0][1])
        fin_result.append(predict_text_2(processed, model, tags))
    return fin_result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    while True:
        choice = input()
        if int(choice) == 1:
            # dataset = 'kaggle/input/EWNERTC_part1.csv'
            dataset = 'kaggle/input/covid_subset_3perc.csv'
            # dataset = 'kaggle/input/ner_dataset.csv'

            x_train, x_test, y_train, y_test, word2idx, tag2idx, max_len, words, tags = functions.preprocess_dataset(
                dataset,
                'word',
                'sentence',
                'entity',
                'Other')


            # file_path = 'kaggle/embed/glove.6B.100d.txt'
            #
            # glove_embeddings = functions.build_matrix_embeddings(path=file_path,
            #                                                      num_tokens=len(words),
            #                                                      embedding_dim=100,
            #                                                      word_index=word2idx)

            functions.ner_2(x_train, x_test, y_train, y_test, word2idx, tag2idx, max_len, words, tags, None)
        elif int(choice) == 2:
            model = load_model('models/my_model2/model.h5')
            # ----------------------------#
            with open('models/my_model2/tags.pickle', 'rb') as handle:
                tags = pickle.load(handle)

            with open('models/my_model2/words.pickle', 'rb') as handle:
                word2idx = pickle.load(handle)

            # ----------------------------#

            print(tags)

            preprocessed = preprocess_text(texts[0], word2idx, model.layers[0].output_shape[0][1])
            # print(preprocessed)
            predict_text(preprocessed, model, tags)
        elif int(choice) == 3:
            texts = []
            with zipfile.ZipFile("data/articles_2023-02-04_500.zip", "r") as f:
                # with zipfile.ZipFile("data/articles_2023-01-07_2000.zip", "r") as f:
                total_f = len(f.namelist())
                counter = 1
                for filename in f.namelist():
                    counter += 1
                    process_file(f, filename, texts)
                f.close()
        elif int(choice) == 4:
            model = load_model('models/my_model2/model.h5')
            # ----------------------------#
            with open('models/my_model2/tags.pickle', 'rb') as handle:
                tags = pickle.load(handle)

            with open('models/my_model2/words.pickle', 'rb') as handle:
                word2idx = pickle.load(handle)

            # ----------------------------#

            result = get_labeled_words(model, texts, word2idx)
            print(result)
        elif int(choice) == 5:
            plot_history_pickle('models/my_model4/history.pickle')
            plot_history_pickle('models/my_model5/history.pickle')
        else:
            break

    # plot_history_pickle('models/my_model4/history.pickle')

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
