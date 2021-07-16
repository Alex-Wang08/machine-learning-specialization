"""locality sensitive hashing"""
import numpy as np
import turicreate
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import paired_distances
from copy import copy
import matplotlib.pyplot as plt

wiki = turicreate.SFrame('people_wiki.sframe')
wiki = wiki.add_row_number()

wiki['tf_idf'] = turicreate.text_analytics.tf_idf(wiki['text'])


def sframe_to_scipy(column):
    x = turicreate.SFrame({'X1': column})
    x = x.add_row_number()
    x = x.stack()