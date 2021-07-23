import turicreate
import numpy as np
import matplotlib.pyplot as plt

wiki = turicreate.SFrame('people_wiki.sframe')
wiki_docs = turicreate.text_analytics.count_words(wiki['text'])