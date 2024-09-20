from distributional import vocab_v_dict, vocab_vc_dict
from collections import defaultdict

corpus_file = 'data/wiki-1percent.txt'
idf_matrix = [[0] * len(vocab_vc_dict) for _ in range(len(vocab_v_dict))]
idf_matrix_3 = [[0] * len(vocab_vc_dict) for _ in range(len(vocab_v_dict))]
idf_matrix_6 = [[0] * len(vocab_vc_dict) for _ in range(len(vocab_v_dict))]

idf_matrix_ = [[0] * len(vocab_v_dict) for _ in range(len(vocab_v_dict))]
idf_matrix_3_ = [[0] * len(vocab_v_dict) for _ in range(len(vocab_v_dict))]
idf_matrix_6_ = [[0] * len(vocab_v_dict) for _ in range(len(vocab_v_dict))]

def compute_idf_factor(corpus_file, vocab_vc_dict):
    word_in_sentence_count = defaultdict(int)
    total_sentences = 0

    with open(corpus_file, 'r') as file:
        for line in file:
            total_sentences += 1
            words_in_sentence = set(line.strip().split())
            for word in words_in_sentence:
                if vocab_vc_dict.get(word) is not None:
                    word_in_sentence_count[word] += 1

    # Compute the IDF factor for each word in VC
    idf_factors = {}
    for word, count in word_in_sentence_count.items():
        idf_factors[word] = total_sentences / count

    return idf_factors, total_sentences

def idf(corpus_file, vocab_v_dict, vocab_vc_dict, co_occurrence_matrix, window_size=3):
    idf_factors, total_sentences = compute_idf_factor(corpus_file, vocab_vc_dict)

    with open(corpus_file, 'r') as file:
        for line in file:
            words = line.strip().split()

            for i, word in enumerate(words):
                x_idx = vocab_v_dict.get(word)

                # If word is in V
                if x_idx is not None:
                    left_window = max(0, i - window_size)
                    right_window = min(len(words), i + window_size + 1)

                    for j in range(left_window, right_window):
                        if i != j:
                            context_word = words[j]
                            y_idx = vocab_vc_dict.get(context_word)

                            # If context word is in VC
                            if y_idx is not None:
                                co_occurrence_matrix[x_idx][y_idx] += 1 * idf_factors[context_word]


# Update the co-occurrence matrix using IDF
idf(corpus_file, vocab_v_dict, vocab_vc_dict, idf_matrix, 1)
idf(corpus_file, vocab_v_dict, vocab_vc_dict, idf_matrix_3, 3)
idf(corpus_file, vocab_v_dict, vocab_vc_dict, idf_matrix_6, 6)

idf(corpus_file, vocab_v_dict, vocab_v_dict, idf_matrix_, 1)
idf(corpus_file, vocab_v_dict, vocab_v_dict, idf_matrix_3_, 3)
idf(corpus_file, vocab_v_dict, vocab_v_dict, idf_matrix_6_, 6)