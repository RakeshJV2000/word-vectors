from distributional import co_occurrence_matrix_3, vocab_v_dict
from idf import co_occurrence_matrix
from pmi import pmi_matrix
import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr

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

        print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
        return spearman_corr


print("Distributional Counting - Window size(w): 3")
print("Men file :")
EvalWS(men_file, vocab_v_dict, co_occurrence_matrix_3)
print("Sim file :")
EvalWS(sim_file, vocab_v_dict, co_occurrence_matrix_3)

print("Inverse Document Frequency - Window size(w): 3")
print("Men file :")
EvalWS(men_file, vocab_v_dict, co_occurrence_matrix)
print("Sim file :")
EvalWS(sim_file, vocab_v_dict, co_occurrence_matrix)

print("Pointwise Mutual Information - Window size(w): 3")
print("Men file :")
EvalWS(men_file, vocab_v_dict, pmi_matrix)
print("Sim file :")
EvalWS(sim_file, vocab_v_dict, pmi_matrix)