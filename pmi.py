from distributional import co_occurrence_matrix_3, vocab_v_dict, vocab_vc_dict
import numpy as np
import math

def compute_pmi_matrix(co_occurrence_matrix, vocab_v_dict, vocab_vc_dict):
    N = np.sum(co_occurrence_matrix)
    p_x = np.sum(co_occurrence_matrix, axis=1) / N
    p_y = np.sum(co_occurrence_matrix, axis=0) / N
    pmi_matrix = np.zeros_like(co_occurrence_matrix, dtype=float)


    for x_idx, word_x in enumerate(vocab_v_dict):
        for y_idx, word_y in enumerate(vocab_vc_dict):
            co_occurrence_count = co_occurrence_matrix[x_idx][y_idx]
            if co_occurrence_count > 0:
                p_xy = co_occurrence_count / N

                # PMI formula: log2(p(X,Y) / (p(X) * p(Y)))
                pmi_value = math.log2(p_xy / (p_x[x_idx] * p_y[y_idx]))
                pmi_matrix[x_idx][y_idx] = pmi_value

    return pmi_matrix

pmi_matrix = compute_pmi_matrix(co_occurrence_matrix_3, vocab_v_dict, vocab_vc_dict)

center_word_idx = vocab_v_dict["coffee"]
pmi_values_for_center_word = pmi_matrix[center_word_idx]
context_pmi_pairs = [(word, pmi_values_for_center_word[idx]) for word, idx in vocab_vc_dict.items()]

sorted_by_pmi = sorted(context_pmi_pairs, key=lambda x: x[1])
print(f"\nTop {10} with the largest PMIs for word coffee:")
for word, pmi_value in reversed(sorted_by_pmi[-10:]):
    print(f"{word}: {pmi_value:.4f}")

print("\n")
print(f"Top {10} with the smallest PMIs for word coffee:")
for word, pmi_value in sorted_by_pmi[:10]:
        print(f"{word}: {pmi_value:.4f}")
