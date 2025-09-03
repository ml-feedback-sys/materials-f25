import os
import urllib.request
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

DATA_DIR = '../../../data/'
filename = "bigrams.json"
url = "https://gist.githubusercontent.com/lydell/259ab9f2ddaa1a64e6bd/raw/6e385151fd5de34e924a1e65f78d152c86afff76/bigrams-all.json"

if not os.path.exists(DATA_DIR+filename):
    print("Downloading bigram data...")
    urllib.request.urlretrieve(url, filename)

with open(filename, "r") as f:
    bigram_counts = json.load(f)

for (a, b), count in bigram_counts:
    print(a,b,count)

# Build the transition matrix
def build_transition_matrix(bigram_counts):
    states = sorted(set([a for (a, b), count in bigram_counts] + [b for (a, b), count in bigram_counts]))
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))

    for (a, b), count in bigram_counts:
        matrix[index[a], index[b]] += count

    row_sums = matrix.sum(axis=1, keepdims=True)
    transition_matrix = matrix / row_sums
    return transition_matrix, states

transition_matrix, states = build_transition_matrix(bigram_counts)

# Sample from the Markov chain
def sample_sequence(transition_matrix, states, start_state, length=100):
    index = {state: i for i, state in enumerate(states)}
    current = index[start_state]
    sequence = [start_state]

    for _ in range(length - 1):
        probs = transition_matrix[current]
        next_state = np.random.choice(states, p=probs)
        sequence.append(next_state)
        current = index[next_state]

    return sequence

sequence = sample_sequence(transition_matrix, states, start_state='a', length=1000)

print('Generated Sequence:')
print(''.join(sequence))

# Compute stationary distribution using matrix decomposition
def compute_stationary_distribution_eigen(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmax(np.abs(eigenvalues))
    assert np.isclose(eigenvalues[idx],1)
    stationary = np.real(eigenvectors[:, idx])
    stationary /= stationary.sum()
    return stationary, eigenvalues

stationary_dist, _ = compute_stationary_distribution_eigen(transition_matrix)

# Compare empirical frequencies
def empirical_frequencies(sequence):
    state_freq = np.zeros(len(states))
    for state in sequence:
        state_freq[np.where([s == state for s in states])[0][0]] += 1
    total = sum(state_freq)
    return state_freq / total

empirical = empirical_frequencies(sequence)
print("\nStationary vs. Empirical Frequencies:")
for state, empirical, prob in zip(states,empirical, stationary_dist):
    print(f"{state}: {empirical:.4f} ({prob:.4f})")