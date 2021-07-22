import turicreate
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.sparse import csr_matrix

wiki = turicreate.SFrame('people_wiki.sframe')
wiki['tf_idf'] = turicreate.text_analytics.tf_idf(wiki['text'])


def sframe_to_scipy(x, column_name):
    assert x[column_name].dtype == dict
    x = x.add_row_number()
    x = x.stack(column_name, ['feature', 'value'])
    f = turicreate.feature