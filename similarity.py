from distributional import co_occurrence_matrix_3, co_occurrence_matrix_6, vocab_v_dict
import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr

men_file = 'data/men.txt'
sim_file = "data/simlex-999.txt"

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.

    Args:
    vec1 (numpy array): First word vector.
    vec2 (numpy array): Second word vector.

    Returns:
    float: The cosine similarity between vec1 and vec2.
    """
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def evaluate_men_similarity(men_file, vocab_v_dict, word_vectors):
    """
    Evaluates word similarity on the MEN dataset using generated word vectors.

    Args:
    men_file (str): Path to the MEN dataset (TSV file).
    vocab_v_dict (dict): Dictionary of target vocabulary (V) with word indices.
    vocab_vc_dict (dict): Dictionary of context vocabulary (VC) with word indices.
    word_vectors (numpy array): 2D array where each row is a word vector.

    Returns:
    int: spearman's similarity scores.
    """

    human_scores = []
    model_scores = []

    with open(men_file, 'r') as file:
        for line in file:
            # Split the line by tabs (TSV format)
            word1, word2, human_score = line.strip().split('\t')
            human_score = float(human_score)

            # Get the word indices for both word1 and word2
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

            # Print the word pair, human score, and model-computed similarity
            # print(f"Word Pair: ({word1},{word2}) | Human Score: {human_score} | Model Similarity: {model_similarity:.4f}")
            human_scores.append(human_score)
            model_scores.append(model_similarity)

        # Compute Spearman's rank correlation between human and model scores
        spearman_corr, _ = spearmanr(human_scores, model_scores)

        print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
        return spearman_corr


print("Distributional Counting - Window size(w): 3")
evaluate_men_similarity(men_file, vocab_v_dict, co_occurrence_matrix_3)
evaluate_men_similarity(sim_file, vocab_v_dict, co_occurrence_matrix_3)
print("Distributional Counting - Window size(w): 3")
evaluate_men_similarity(men_file, vocab_v_dict, co_occurrence_matrix_6)
evaluate_men_similarity(sim_file, vocab_v_dict, co_occurrence_matrix_6)
