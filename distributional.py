def remove_duplicates_preserve_order(word_list):
    seen = set()
    unique_list = []
    for word in word_list:
        if word not in seen:
            unique_list.append(word)
            seen.add(word)
    return unique_list

with open('data/vocab-15kws.txt', 'r') as file:
    v_list = [line.strip() for line in file.readlines()]
    v_list = remove_duplicates_preserve_order(v_list)

with open('data/vocab-5k.txt', 'r') as file:
    vc_list = [line.strip() for line in file.readlines()]
    vc_list = remove_duplicates_preserve_order(vc_list)

# Create dictionaries that map words to their indices
vocab_v_dict = {word: idx for idx, word in enumerate(v_list)}
vocab_vc_dict = {word: idx for idx, word in enumerate(vc_list)}

# Initialize the co-occurrence matrix of size len(V) x len(VC)
co_occurrence_matrix_3 = [[0] * len(vc_list) for _ in range(len(v_list))]
co_occurrence_matrix_6 = [[0] * len(vc_list) for _ in range(len(v_list))]


def update_co_occurrence_matrix(corpus_file, vocab_v_dict, vocab_vc_dict, co_occurrence_matrix, window_size):
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
                                co_occurrence_matrix[x_idx][y_idx] += 1


corpus_file = 'data/wiki-1percent.txt'
update_co_occurrence_matrix(corpus_file, vocab_v_dict, vocab_vc_dict, co_occurrence_matrix_3, window_size=3)
update_co_occurrence_matrix(corpus_file, vocab_v_dict, vocab_vc_dict, co_occurrence_matrix_6, window_size=6)

print("word pair\t\tw=3\tw=6")
print("(chicken, the)\t\t",
      co_occurrence_matrix_3[vocab_v_dict['chicken']][vocab_vc_dict['the']], "\t",
      co_occurrence_matrix_6[vocab_v_dict['chicken']][vocab_vc_dict['the']]
      )
print("(chicken, wings)\t",
      co_occurrence_matrix_3[vocab_v_dict['chicken']][vocab_vc_dict['wings']], "\t",
      co_occurrence_matrix_6[vocab_v_dict['chicken']][vocab_vc_dict['wings']]
      )
print("(chicago, chicago)\t",
      co_occurrence_matrix_3[vocab_v_dict['chicago']][vocab_vc_dict['chicago']], "\t",
      co_occurrence_matrix_6[vocab_v_dict['chicago']][vocab_vc_dict['chicago']]
      )
print("(coffee, the)\t\t",
      co_occurrence_matrix_3[vocab_v_dict['coffee']][vocab_vc_dict['the']], "\t",
      co_occurrence_matrix_6[vocab_v_dict['coffee']][vocab_vc_dict['the']]
      )
print("(coffee, cup)\t\t",
      co_occurrence_matrix_3[vocab_v_dict['coffee']][vocab_vc_dict['cup']], "\t",
      co_occurrence_matrix_6[vocab_v_dict['coffee']][vocab_vc_dict['cup']]
      )
print("(coffee, coffee)\t",
      co_occurrence_matrix_3[vocab_v_dict['coffee']][vocab_vc_dict['coffee']], "\t",
      co_occurrence_matrix_6[vocab_v_dict['coffee']][vocab_vc_dict['coffee']]
      )