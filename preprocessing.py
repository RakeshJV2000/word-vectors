import numpy as np
import pandas as pd
import re

vocab_list = []
pattern = re.compile(r'[^a-zA-Z]+')
with open('data/vocab-15kws.txt', 'r') as file:
    for line in file:
        word = re.sub(pattern, '', line.strip())

        # Only add non-empty words to the list
        if word:
            vocab_list.append(word)

with open('data/cleaned_vocab-15kws.txt', 'w') as output_file:
    for word in vocab_list:
        output_file.write(f"{word}\n")

vocab_list = []
with open('data/vocab-5k.txt', 'r') as file:
    for line in file:
        word = re.sub(pattern, '', line.strip())

        # Only add non-empty words to the list
        if word:
            vocab_list.append(word)

with open('data/cleaned_vocab-5k.txt', 'w') as output_file:
    for word in vocab_list:
        output_file.write(f"{word}\n")
