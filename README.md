# word-vectors

In this project, different methods for building word vectors were implemented using a Wikipedia corpus. A vocabulary 𝑉 and a context vocabulary 𝑉𝐶 were defined to calculate word vectors. Word co-occurrences within a specified context window size were counted, and word vectors were constructed based on these co-occurrence counts. Various techniques, including **Distributional Counts**, **TF-IDF**, and **Pointwise Mutual Information (PMI)**, were employed to compute the word vectors.

The evaluation of the word vectors was conducted by comparing the cosine similarity between word pairs to manually annotated similarity scores in the provided MEN and SimLex-999 datasets. **Spearman’s rank correlation** coefficient was used as the evaluation metric to measure the alignment between the computed word vector similarities and human-annotated scores. The analysis explored the effect of varying context window sizes and vocabulary on word vector performance. Additionally, words with multiple senses were examined to assess how the constructed vectors capture different meanings.

## Distributional Counting

Distributional counting quantifies how often words co-occur within specific contextual windows in a given corpus. The basic idea is that words that appear in similar contexts tend to have similar meanings. In this context, the task involves counting how many times a context word 𝑦 appears within a context window of size 𝑤, centered around a target word 𝑥, using a sample corpus.

## Inverse Document Frequency(IDF)

Inverse Document Frequency quantifies the importance of a word within a collection of documents. It helps distinguish common words from rare ones by assigning higher weights to terms that appear in fewer documents, making them more significant in identifying the uniqueness of content. IDF is typically used alongside Term Frequency (TF) to form the TF-IDF weighting scheme, which balances how often a word appears in a document (TF) against how rare or frequent it is across all documents.

## Pointwise Mutual Information(PMI)

Pointwise Mutual Information (PMI) is a measure used to quantify the strength of association between two words based on their co-occurrence in a corpus. PMI compares the actual co-occurrence of two words with how often we would expect them to co-occur by chance, providing insight into how strongly they are related. A high PMI value indicates that two words co-occur more often than expected by chance, suggesting a strong association.

## Quantitative Comparisons
The results indicate that PMI consistently performs better than the other two methods across all window sizes, showing strong correlations in both small and large windows. TF-IDF improves significantly with larger window sizes, performing well, especially for window size 𝑤 = 6, where it shows the highest correlation among all methods. Distributional Counts, on the other hand, consistently underperform compared to both PMI and TF-IDF, with only modest improvements as window size increases. Overall, PMI and TF-IDF are more effective for building word vectors, particularly as context windows expand, while Distributional Counts remain the weakest method in comparison.

### Window size:
As the window size increases, the correlation for word pairs in the MEN file tends to improve performance. In SimLex, the correlation decreases slightly with larger window sizes, showing that a larger window may capture less relevant relationships for fine-grained similarity judgments.

### Context Vocabulary:
For Distributional Counts, the correlation remains relatively stable when using a larger context vocabulary. Regarding IDF, the larger context vocabulary boosts performance suggesting that TF-IDF benefits from more words in the language to compute more refined document frequencies. And for PMI, the larger context vocabulary results in a slight increase in performance for MEN likely because PMI uses the entire set of word pairings and benefits from the increased variety of co-occurrence possibilities in a larger vocabulary.

### Observations:
The is an opposite trend seen between the MEN file and the SIMlex file as the w increases. Possibly because the similarity score in MEN reflects the more general, broad word similarities (e.g., "cat" and "feline," "automobile" and "car"), which are more likely to be captured by context windows that are preferably larger. Whereas SIMlex focuses on more nuanced word similarities (e.g., "smart" and "intelligent"), which may not always align with raw co-occurrence patterns and hence have lower correlations. For all methods, SimLex achieves lower correlations, suggesting that the datasets measure different aspects of word similarity. SimLex is more challenging for these methods due to its emphasis on fine-grained conceptual similarity rather than broad semantic relatedness.

Concerning context vocabulary, we see all the methods had slightly better correlation scores. It is because as we add more context words, we are more likely to capture better contextual meaning for a given word. A larger context vocabulary (vocab-15kws.txt) generally improves results, especially for TF-IDF and PMI, as more context words contribute to better word representations.

## Qualitative Analysis

### Do nearest neighbors tend to have the same part-of-speech tag as the query word?
For verbs, there seems to be a stronger tendency for smaller windows to find neighbors of the same POS, while larger windows introduce more nouns and semantically related concepts. For adjectives, smaller windows also include neighbors of the same POS, but there’s more variety even in the smaller window size, with some nouns and verbs appearing. This suggests that adjectives may have looser syntactic constraints and more varied neighbors compared to verbs.

### Multi sense words
Multi-sense words like "bank", "apple", and "light" show clear distinctions between their different meanings depending on the window size. Smaller windows tend to focus on one specific sense (often the more common or concrete sense) and capture more immediate, syntactic relationships, focusing on the dominant or primary sense of the word., while larger windows introduce neighbors that reflect multiple senses and introduce more semantic relationships, bringing in terms that reflect multiple meanings or uses of the word.