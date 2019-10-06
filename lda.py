#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
#https://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/Gensim%20Newsgroup.ipynb
#python -m spacy download es_core_news_sm
#https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
from nltk.corpus import stopwords
import spacy
import os
import csv
import sys
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

nlp = spacy.load('es_core_news_sm')

#cluster victor
def load_stopwords(string):
    """
    Lee archivos .txt con stopwords. Deben estar separadas por salto de linea
    :param string:
    :return:
    """
    print("Reading Stopwords")
    # Si el archivo existe
    if os.path.isfile(string):
        with open(string, 'r', encoding="utf8") as words:
            sw = words.read().split('\n')
    # Si el nombre es english o spanish
    else:
        sw = stopwords.words(string)
    return sw
def lda(president):
    stop_words = load_stopwords('stopwords_es_test.txt')

    content = []
    # https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
    data = []
    with open('data/csv/union.csv') as csvfile:
        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.

            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        readCSV = csv.DictReader(csvfile)
        for row in readCSV:
            if row['president'] == president:
                content = str(row['content']).lower()
                data.append(content)
    print(data[0])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    # nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus[:1])

    print(id2word[0])
    # corpora.MmCorpus.serialize('newsgroups.mm', corpus)
    # id2word.save('newsgroups.dict')
    print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # lda_model.save('newsgroups_50_lda.model')
    print(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, president + '.html')
presidents = ['pinera', 'bachelet', 'allende', 'macri', 'kirchner', 'fernandez']
for president in presidents:
    lda(president)