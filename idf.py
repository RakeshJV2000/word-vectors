from distributional import co_occurrence_matrix_3, vocab_v_dict, vocab_vc_dict
from collections import defaultdict

corpus_file = 'data/wiki-1percent.txt'
co_occurrence_matrix = [[0] * len(vocab_vc_dict) for _ in range(len(vocab_v_dict))]

def compute_idf_factor(corpus_file, vocab_vc_dict):
    """
    Computes the IDF factor for each context word (y) in the vocabulary.

    Args:
    corpus_file (str): Path to the wiki-1percent.txt corpus file.
    vocab_vc_dict (dict): Context vocabulary (VC) with word indices.

    Returns:
    dict: A dictionary where keys are words and values are the TDF factors.
    """
    # Dictionary to count how many sentences contain each word (y)
    word_in_sentence_count = defaultdict(int)
    total_sentences = 0

    with open(corpus_file, 'r') as file:
        for line in file:
            total_sentences += 1
            words_in_sentence = set(line.strip().split())  # Unique words in the sentence

            # For each word in the sentence, increment the count
            for word in words_in_sentence:
                if vocab_vc_dict.get(word) is not None:
                    word_in_sentence_count[word] += 1

    # Compute the TDF factor for each word in VC
    idf_factors = {}
    for word, count in word_in_sentence_count.items():
        idf_factors[word] = total_sentences / count

    return idf_factors, total_sentences

def idf(corpus_file, vocab_v_dict, vocab_vc_dict, co_occurrence_matrix, window_size=3):
    """
    Updates the co-occurrence matrix using the IDF formula.

    Args:
    corpus_file (str): Path to the wiki-1percent.txt corpus file.
    vocab_v_dict (dict): Dictionary of target vocabulary (V) with word indices.
    vocab_vc_dict (dict): Dictionary of context vocabulary (VC) with word indices.
    co_occurrence_matrix (list of lists): The co-occurrence matrix to be updated.
    window_size (int): The size of the context window (w).

    Returns:
    None: The co-occurrence matrix is updated in place.
    """
    # Compute the TDF factors for each word in VC
    idf_factors, total_sentences = compute_idf_factor(corpus_file, vocab_vc_dict)

    # Read the corpus file line by line (each line is a sentence)
    with open(corpus_file, 'r') as file:
        for line in file:
            # Split the line into words (assuming space-separated words)
            words = line.strip().split()

            # Iterate through each word in the sentence
            for i, word in enumerate(words):
                # Get the index of the target word in vocab_v_dict (if exists)
                x_idx = vocab_v_dict.get(word)

                # If word is in vocab_v_dict (V)
                if x_idx is not None:
                    # Define the window range
                    left_window = max(0, i - window_size)
                    right_window = min(len(words), i + window_size + 1)

                    # Iterate over the context window
                    for j in range(left_window, right_window):
                        if i != j:  # Ensure the context word is not the center word
                            context_word = words[j]

                            # Get the index of the context word in vocab_vc_dict (if exists)
                            y_idx = vocab_vc_dict.get(context_word)

                            # If context word is in vocab_vc_dict (VC)
                            if y_idx is not None:
                                # Increment the raw co-occurrence count
                                co_occurrence_matrix[x_idx][y_idx] += 1 * idf_factors[context_word]


# Update the co-occurrence matrix using IDF
idf(corpus_file, vocab_v_dict, vocab_vc_dict, co_occurrence_matrix, 3)