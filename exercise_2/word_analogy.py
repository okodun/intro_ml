# import random
import numpy as np

vocabulary_file = "word_embeddings.txt"
N = 2


# Read words
print("Read words...")
with open(vocabulary_file, "r") as f:
    words = [x.rstrip().split(" ")[0] for x in f.readlines()]

# Read word vectors
print("Read word vectors...")
with open(vocabulary_file, "r") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(" ")
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print("Vocabulary size")
print(len(vocab))
print(vocab["man"])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print("Vocabulary word vectors")
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == "<unk>":
        continue
    W[vocab[word], :] = v
print(W.shape)

# Main loop for analogy
while True:
    input_term = input("\nEnter word associations (EXIT to break): ")
    if input_term == "EXIT":
        break
    else:
        terms = input_term.split("-")
        if len(terms) != 3:
            print("Invalid input!")
            continue
        try:
            word_vecs = [W[vocab[term.lower()]] for term in terms]
        except KeyError:
            print("Search string contains unknown words!")
            continue
        search_vec = word_vecs[2] + (word_vecs[1] - word_vecs[0])
        distances = np.sum(np.square(W - search_vec), axis=1)

        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for idx in np.argpartition(distances, N)[:N]:
            print("%35s\t\t%f\n" % (ivocab[idx], distances[idx]))
