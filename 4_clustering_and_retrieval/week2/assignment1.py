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

"""Q3"""


def get_common_words(name1, name2, num_of_words=10):
    words1 = top_words(name1)
    words2 = top_words(name2)
    top_combined_words = words1.join(words2, on='word')
    return top_combined_words.sort('count', ascending=False)[0:num_of_words]


combined_words_obama_bush = get_common_words('Barack Obama', 'George W. Bush', 10)

"""
Q4
Among the words that appear in both Barack Obama and Phil Schiliro, 
take the 5 that have largest weights in Obama. 
How many of the articles in the Wikipedia dataset contain all of those 5 words?
"""
wiki['tf_idf'] = turicreate.text_analytics.tf_idf(wiki['word_count'])
model_tf_idf = turicreate.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                   method='brute_force', distance='euclidean')

model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)


def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word', 'weight'])
    return word_count_table.sort('weight', ascending=False)


obama_tf_idf = top_words_tf_idf('Barack Obama')
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
combined_tf_idf = obama_tf_idf.join(schiliro_tf_idf, on='word').sort('weight', ascending=False)

common_words_obama_schiliro = combined_tf_idf['word'][0:5]


def has_top_words(word_count_vector):
    unique_words = set(word_count_vector.keys())
    return set(common_words_obama_schiliro).issubset(unique_words)


wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
total1 = wiki['has_top_words'].sum()


"""
Q5
Compute the Euclidean distance between TF-IDF features of Obama and Biden.
Round your answer to 3 decimal places. Use American-style decimals (e.g. 110.921).
"""
Name_Obama = 'Barack Obama'
Name_Biden = 'Joe Biden'
distance_obama_biden_tf_idf = turicreate.distances.euclidean(wiki['tf_idf'][wiki['name'] == Name_Obama][0],
                                                             wiki['tf_idf'][wiki['name'] == Name_Biden][0])

