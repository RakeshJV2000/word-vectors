import time
start_time = time.time()
from distributional import (count_matrix_3, vocab_v_dict, count_matrix_6,
                            count_matrix, count_matrix_, count_matrix_6_, count_matrix_3_)
from idf import idf_matrix_3, idf_matrix, idf_matrix_6, idf_matrix_, idf_matrix_3_, idf_matrix_6_
from pmi import pmi_matrix_3, pmi_matrix, pmi_matrix_6, pmi_matrix_, pmi_matrix_3_, pmi_matrix_6_
import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr
import spacy
nlp = spacy.load('en_core_web_sm')

men_file = 'data/men.txt'
sim_file = "data/simlex-999.txt"

def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def EvalWS(files, vocab_v_dict, word_vectors):
    human_scores = []
    model_scores = []

    with open(files, 'r') as file:
        for line in file:
            word1, word2, human_score = line.strip().split('\t')
            human_score = float(human_score)

            idx1 = vocab_v_dict.get(word1)
            idx2 = vocab_v_dict.get(word2)

            # If both words are in the vocabulary, compute cosine similarity
            if idx1 is not None and idx2 is not None:
                vec1 = word_vectors[idx1]
                vec2 = word_vectors[idx2]
                model_similarity = cosine_similarity(vec1, vec2)
            else:
                # If either word is not found, set similarity to 0.0
                model_similarity = 0.0

            human_scores.append(human_score)
            model_scores.append(model_similarity)

        spearman_corr, _ = spearmanr(human_scores, model_scores)

        # print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
        return float(f"{spearman_corr:.4f}")

print("\nEvalWS for Distributional counting with w=3")
x = EvalWS(men_file, vocab_v_dict, count_matrix_3)
print(f"Spearman's Rank Correlation for MEN file: {x}")
y = EvalWS(sim_file, vocab_v_dict, count_matrix_3)
print(f"Spearman's Rank Correlation for SIMlex file: {y}")

print("\nSpearman's Rank Correlation for different context window size and context vocabulary")
print("\n\t\t w=1\t\tw=3\t\tw=6")
print("Distributional")
print("MEN file\t",EvalWS(men_file, vocab_v_dict, count_matrix),
      "\t",x,
      "\t",EvalWS(men_file, vocab_v_dict, count_matrix_6))

print("SIMlex file\t",EvalWS(sim_file, vocab_v_dict, count_matrix),
      "\t",y,
      "\t",EvalWS(sim_file, vocab_v_dict, count_matrix_6))

print("TF-IDF")
print("MEN file\t",EvalWS(men_file, vocab_v_dict, idf_matrix),
      "\t",EvalWS(men_file, vocab_v_dict, idf_matrix_3),
      "\t",EvalWS(men_file, vocab_v_dict, idf_matrix_6))

print("SIMlex file\t",EvalWS(sim_file, vocab_v_dict, idf_matrix),
      "\t",EvalWS(sim_file, vocab_v_dict, idf_matrix_3),
      "\t",EvalWS(sim_file, vocab_v_dict, idf_matrix_6))

print("PMI")
print("MEN file\t",EvalWS(men_file, vocab_v_dict, pmi_matrix),
      "\t",EvalWS(men_file, vocab_v_dict, pmi_matrix_3),
      "\t",EvalWS(men_file, vocab_v_dict, pmi_matrix_6))

print("SIMlex file\t",EvalWS(sim_file, vocab_v_dict, pmi_matrix),
      "\t",EvalWS(sim_file, vocab_v_dict, pmi_matrix_3),
      "\t",EvalWS(sim_file, vocab_v_dict, pmi_matrix_6))

print("\nThe below is after using vocab-15kws.txt for Vc")
print("Distributional")
print("MEN file\t",EvalWS(men_file, vocab_v_dict, count_matrix_),
      "\t",EvalWS(men_file, vocab_v_dict, count_matrix_3_),
      "\t",EvalWS(men_file, vocab_v_dict, count_matrix_6_))

print("SIMlex file\t",EvalWS(sim_file, vocab_v_dict, count_matrix_),
      "\t\t",EvalWS(sim_file, vocab_v_dict, count_matrix_3_),
      "\t",EvalWS(sim_file, vocab_v_dict, count_matrix_6_))

print("TF-IDF")
print("MEN file\t",EvalWS(men_file, vocab_v_dict, idf_matrix_),
      "\t",EvalWS(men_file, vocab_v_dict, idf_matrix_3_),
      "\t\t",EvalWS(men_file, vocab_v_dict, idf_matrix_6_))

print("SIMlex file\t",EvalWS(sim_file, vocab_v_dict, idf_matrix_),
      "\t",EvalWS(sim_file, vocab_v_dict, idf_matrix_3_),
      "\t",EvalWS(sim_file, vocab_v_dict, idf_matrix_6_))

print("PMI")
print("MEN file\t",EvalWS(men_file, vocab_v_dict, pmi_matrix_),
      "\t",EvalWS(men_file, vocab_v_dict, pmi_matrix_3_),
      "\t",EvalWS(men_file, vocab_v_dict, pmi_matrix_6_))

print("SIMlex file\t",EvalWS(sim_file, vocab_v_dict, pmi_matrix_),
      "\t",EvalWS(sim_file, vocab_v_dict, pmi_matrix_3_),
      "\t",EvalWS(sim_file, vocab_v_dict, pmi_matrix_6_))


def find_nearest_neighbors(query_word, vocab_v_dict, pmi_matrix, k=10):
    if query_word not in vocab_v_dict:
        print(f"Query word '{query_word}' not found in vocabulary.")
        return []

    query_idx = vocab_v_dict[query_word]
    query_vector = pmi_matrix[query_idx]

    similarities = []
    for word, idx in vocab_v_dict.items():
        if word != query_word:  # Omit the query word itself
            similarity = cosine_similarity(query_vector, pmi_matrix[idx])
            similarities.append((word, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

query_word = 'judges'
print(f"\n10 nearest neighbors for '{query_word}' with window size w = 1:")
neighbors_w1 = find_nearest_neighbors(query_word, vocab_v_dict, pmi_matrix_)
for word, sim in neighbors_w1:
    print(f"Word: {word}, Cosine Similarity: {sim:.4f}")

print(f"\n10 nearest neighbors for '{query_word}' with window size w = 6:")
neighbors_w6 = find_nearest_neighbors(query_word, vocab_v_dict, pmi_matrix_6_)
for word, sim in neighbors_w6:
    print(f"Word: {word}, Cosine Similarity: {sim:.4f}")

def get_pos_tag(word):
    doc = nlp(word)
    return doc[0].pos_

def find_nearest_neighbors_with_pos(query_word, vocab_v_dict, pmi_matrix, k=5):
    query_idx = vocab_v_dict[query_word]
    query_vector = pmi_matrix[query_idx]

    similarities = []
    for word, idx in vocab_v_dict.items():
        if word != query_word:
            similarity = cosine_similarity(query_vector, pmi_matrix[idx])
            similarities.append((word, similarity, get_pos_tag(word)))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def analyze_nearest_neighbors(query_word, vocab_v_dict, pmi_matrix_, pmi_matrix_6_):
    query_pos = get_pos_tag(query_word)

    print(f"Query word: '{query_word}' (POS: {query_pos})")

    # For window size w=1
    print(f"\nNearest neighbors for '{query_word}' with window size w=1:")
    neighbors_w1 = find_nearest_neighbors_with_pos(query_word, vocab_v_dict, pmi_matrix_)
    for word, sim, pos in neighbors_w1:
        print(f"Neighbor: {word}, Cosine Similarity: {sim:.4f}, POS: {pos}")

    # For window size w=6
    print(f"\nNearest neighbors for '{query_word}' with window size w=6:")
    neighbors_w6 = find_nearest_neighbors_with_pos(query_word, vocab_v_dict, pmi_matrix_6_)
    for word, sim, pos in neighbors_w6:
        print(f"Neighbor: {word}, Cosine Similarity: {sim:.4f}, POS: {pos}")


query_words = ['produced', 'short', 'in']

for query_word in query_words:
    print("\n=============================")
    analyze_nearest_neighbors(query_word, vocab_v_dict, pmi_matrix_, pmi_matrix_6_)

query_words = ['produced', 'short', 'in']

for query_word in query_words:
    print("\n=============================")
    analyze_nearest_neighbors(query_word, vocab_v_dict, pmi_matrix_, pmi_matrix_6_)

print("Words with multiple senses")
multi_sense_words = ['bank', 'cell', 'apple', 'light', 'well', 'frame']

for query_word in multi_sense_words:
    print("\n=============================")
    analyze_nearest_neighbors(query_word, vocab_v_dict, pmi_matrix_, pmi_matrix_6_)

end_time = time.time()
tot = (end_time - start_time)/60
print(f"\nNo of minutes taken to execute :{((end_time - start_time)/60):.2f}")