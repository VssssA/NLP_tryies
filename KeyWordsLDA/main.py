import numpy as np
import pandas as pd

import spacy
import random
from gensim import models, corpora, similarities
from wordcloud import WordCloud

from spacy.lang.ru.examples import sentences
from spacy.lang.ru import Russian
from spacy import load

nlp = spacy.load("ru_core_news_sm")

# Убираем ner и parser для ускорения процесса
unwanted_pipes = ["ner", "parser"]

# собственно сам токенизатор
def custom_tokenizer(doc):
    with nlp.disable_pipes(*unwanted_pipes):
        return [t.lemma_ for t in nlp(doc) if t.is_alpha and not t.is_space and not t.is_punct and not t.is_stop and t.pos_ in ["ADJ","NOUN","VERB"]]

# new_text = "Компания УЦСБ – российский системный интегратор. Более 15 лет мы занимаемся созданием и модернизацией основных составляющих ИТ-систем современного предприятия, оказанием услуг в области проектирования, разработкой и внедрением решений по обеспечению информационной безопасности."
# new_tokens = list(map(custom_tokenizer, [nlp(new_text)]))[0]
# new_bow = lda_model25.doc2bow(new_tokens)
#
# print(new_text,'\n')
# get_similar_text(new_bow)

lda_model =  models.LdaModel.load('lda_model25')

lda_index = similarities.MatrixSimilarity(lda_model)

