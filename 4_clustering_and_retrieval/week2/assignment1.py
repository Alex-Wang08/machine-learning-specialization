import turicreate
import numpy as np
import matplotlib.pyplot as plt

wiki = turicreate.SFrame('people_wiki.sframe')
wiki['word_count'] = turicreate.text_analytics.count_words(wiki['text'])

model = turicreate.nearest_neighbors.create(wiki, label='name', features=['word_count'],
                                            method='brute_force', distance='euclidean')

model.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)


def top_words(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word', 'count'])
    return word_count_table.sort('count', ascending=False)


obama_words = top_words('Barack Obama')
barrio_words = top_words('Francisco Barrio')
combined_words = obama_words.join(barrio_words, on='word')
combined_words = combined_words.rename({'count': 'Obama', 'count.1': 'Barrio'})
combined_words.sort('Obama', ascending=False)

common_words = combined_words['word'][0:5]


def has_top_words(word_count_vector):
    unique_words = set(word_count_vector.keys())
    return set(common_words).issubset(unique_words)


wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
"""Q1"""
total = wiki['has_top_words'].sum()

"""Q2"""
obama_words_count = wiki['word_count'][wiki['name'] == 'Barack Obama'][0]
bush_words_count = wiki['word_count'][wiki['name'] == 'George W. Bush'][0]
biden_words_count = wiki['word_count'][wiki['name'] == 'Joe Biden'][0]

distance_obama_bush = turicreate.distances.euclidean(obama_words_count, bush_words_count)
distance_obama_biden = turicreate.distances.euclidean(obama_words_count, biden_words_count)
distance_bush_biden = turicreate.distances.euclidean(bush_words_count, biden_words_count)

