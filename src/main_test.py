
import spacy
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import zipfile
import pickle
from keras.models import load_model
import numpy as np
import re
from nltk.corpus import stopwords

import functions


pattern = re.compile('.*-(MONEY|QUANTITY|PERCENT|ORDINAL|DATE|TIME|CARDINAL)|(Other)')


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens


def process_file(file, filename, texts):
    content = file.read(filename)
    if type(content) == bytes:
        text = content.decode('utf-8')
        texts.append(text)

    if len(content.strip()) == 0:
        print("No text was found")
        return


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


def predict_text_2(text, model, tags):
    ents = []
    labels = []
    for i in range(len(text)):
        p = model.predict(np.array([text[0][i]]), verbose=0)
        p = np.argmax(p, axis=-1)
        for idx, pred in enumerate(p[0][0:len(text[1][i])]):
            if len(word := text[1][i][idx]) > 3:
                ents.append(word)
                if not pattern.match(tags[pred]):
                    # append the same word once more to increase its statistics and weight artificially
                    ents.append(word)
                    labels.append(tags[pred])
    return ents, labels


def get_labeled_words(model, texts, word2idx):
    proc_labels = []
    proc_ents = []
    for text in texts:
        processed = preprocess_text(text, word2idx, model.layers[0].output_shape[0][1])
        ents, labels = predict_text_2(processed, model, tags)
        proc_ents.append(ents)
        proc_labels.append(labels)
    return proc_ents, proc_labels


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    choice = input()
    if int(choice) == 1:
        # Load the NER model
        nlp = spacy.load("en_core_web_sm")

        # Use the model to extract named entities from text
        #text = "Apple is looking at buying U.K. startup for $1 billion"

        texts = []
        with zipfile.ZipFile("data/articles_2021-11-05_1000.zip", "r") as f:
        # with zipfile.ZipFile("data/articles_2023-01-07_2000.zip", "r") as f:
            total_f = len(f.namelist())
            counter = 1
            for filename in f.namelist():
                counter += 1
                process_file(f, filename, texts)
            f.close()

        model = load_model('models/my_model2/model.h5')
        # ----------------------------#
        with open('models/my_model2/tags.pickle', 'rb') as handle:
            tags = pickle.load(handle)

        with open('models/my_model2/words.pickle', 'rb') as handle:
            word2idx = pickle.load(handle)


        proc_ents, proc_labels = get_labeled_words(model, texts, word2idx)


        # for text in texts:
        #     doc = nlp(text)
        #
        #     proc_labels.append([ent.label_ for ent in doc.ents if ent.label_ not in ['CARDINAL', 'DATE', 'TIME', 'ORDINAL', 'PERCENT'] and ent.text.lower() != 'reuters'])
        #     proc_ents.append([ent.text for ent in doc.ents if ent.label_ not in ['CARDINAL', 'DATE', 'TIME', 'ORDINAL', 'PERCENT'] and ent.text.lower() != 'reuters'])
        #
        #     # Print the named entities and their labels
        #     # for ent in doc.ents:
        #     #     print(ent.text, ent.label_)


        #print(proc_ents)
        #print(proc_labels)

        print("Done 1")

        # Create a dictionary mapping named entities to integer ids
        dictionary = Dictionary(proc_ents)

        # Create a document-term matrix where each document is a text and each term is a named entity
        corpus = [dictionary.doc2bow(text) for text in proc_ents]

        # Train the LDA model on the corpus
        #lda_model = LdaModel(corpus, num_topics=6)

        num_topics = 6

        lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=100,
                                                workers=7,
                                                chunksize=10,
                                                passes=20,
                                                alpha='symmetric',
                                                iterations=200,
                                                per_word_topics=False)

        # Infer the topic distribution for each text
        text_topics = [lda_model[c] for c in corpus]

        for i in range(num_topics):
            topic_words = lda_model.show_topic(i)
            tmp = [(word, prob) for word, prob in topic_words]
            print("Topic %d: \n%s\n" % (i, tmp))

        print(text_topics[0:10])
    else:
        functions.get_last_N_articles(1000)
