# %%
# from datasets import load_dataset

# dataset = load_dataset("yelp_review_full")

# %%
import pickle
# with open('yelp_dataset.pkl', 'wb') as file:
#     pickle.dump(dataset, file)

# %%
with open('yelp_dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

# %%
print(dataset)

# %%
train_dataset = dataset['train']
test_dataset = dataset['test']

# Convert train and test datasets to arrays
train_data = train_dataset['text']
train_labels = train_dataset['label']
test_data = test_dataset['text']
test_labels = test_dataset['label']

# Convert labels to lists (optional)
# train_labels = train_labels.tolist()
# test_labels = test_labels.tolist()

# %%
print(train_data[0])
print(train_labels[0])


# %% [markdown]
# ## Preprocess

# %%
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# %%
import sys  
!{sys.executable} -m pip install contractions
import contractions

# %%
# import re  # Import the regular expressions module

# # use tokenizer to remove punctuation

# def tokenize_text(text):
#     # Replace "!" with "exm" using regular expressions
#     text = re.sub(r'!', ' exm', text)
    
#     expanded_words = []
#     for word in text.split():
#         # using contractions.fix to expand the shortened words
#         expanded_words.append(contractions.fix(word))   
    
#     expanded_text = ' '.join(expanded_words)
#     # print(expanded_text)
    
#     text = expanded_text    
    
#     # replace a-b with a and b
#     text = text.replace('-', ' ')
    
#     tokens = nltk.word_tokenize(text)
    
#     # Add an extra occurrence for all-uppercase words with more than one letter
#     # temp = [word if (len(word) > 1 and word.isupper()) else None for word in tokens]
#     temp = []
#     for word in tokens:
#         if len(word) > 1 and word.isupper():
#             temp.append(word)
#     tokens.extend(x for x in temp if x)
    
#     # convert to lower case
#     tokens = [w.lower() for w in tokens]
#     # dr. = dr and st. = st and so on
#     tokens = [w.replace('.', '') for w in tokens]
    
#     # remove punctuation
#     # tokens = [word for word in tokens if word.isalpha()]
    
#     # Remove stop words
#     stop_words = set(stopwords.words("english"))
#     tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
#     return tokens

# # print(train_data[4])
# # tokenize_text(train_data[4])


# %%
# tokens = []
# print(len(train_data))

# for i in range(len(train_data)):
#     tokens.append(tokenize_text(train_data[i]))
#     if i % 1000 == 0:
#         print(i)

# # save to pkl file
# with open('yelp_train_tokens_no_stop.pkl', 'wb') as file:
#     pickle.dump(tokens, file)

# %%
# import pickle
# from collections import Counter

# # Load the tokenized data
# with open('yelp_train_tokens_no_stop.pkl', 'rb') as file:
#     tokens = pickle.load(file)

# # Define the frequency cutoff threshold
# frequency_cutoff = 10

# # Count word frequencies
# word_counts = Counter(word for tokens_list in tokens for word in tokens_list)

# # Filter out words with counts less than the threshold
# filtered_tokens = [[word for word in tokens_list if word_counts[word] >= frequency_cutoff] for tokens_list in tokens]

# # Save the filtered tokens to a new file
# with open('yelp_train_tokens_filtered.pkl', 'wb') as file:
#     pickle.dump(filtered_tokens, file)

# %%
# tokens_test = []
# print(len(test_data))

# for i in range(len(test_data)):
#     tokens_test.append(tokenize_text(test_data[i]))
#     if i % 1000 == 0:
#         print(i)

# with open('yelp_test_tokens_no_stop.pkl', 'wb') as file:
#     pickle.dump(tokens_test, file)

# %%
# import pickle
# from collections import Counter

# # Load the tokenized data
# with open('yelp_test_tokens_no_stop.pkl', 'rb') as file:
#     tokens = pickle.load(file)

# # Define the frequency cutoff threshold
# frequency_cutoff = 10

# # Count word frequencies
# word_counts = Counter(word for tokens_list in tokens for word in tokens_list)

# # Filter out words with counts less than the threshold
# filtered_tokens = [[word for word in tokens_list if word_counts[word] >= frequency_cutoff] for tokens_list in tokens]

# # Save the filtered tokens to a new file
# with open('yelp_test_tokens_filtered.pkl', 'wb') as file:
#     pickle.dump(filtered_tokens, file)

# %%
import numpy as np

train_tokens = []
with open('yelp_train_tokens_filtered.pkl', 'rb') as file:
    train_tokens = pickle.load(file)

test_tokens = []
with open('yelp_test_tokens_filtered.pkl', 'rb') as file:
    test_tokens = pickle.load(file)
    
print("Loaded tokens")



# %%
# in train data replace every 1000th word with UNK randomly

import random

for i in range(len(train_tokens)):
    for j in range(len(train_tokens[i])):
        if random.randint(1, 1000) == 1:
            train_tokens[i][j] = 'UNK'
        

# %%
# Build the BoW representation manually
# Create a vocabulary by collecting unique words from the training data
vocab = set()
for tokens in train_tokens:
    vocab.update(tokens)

# Create a dictionary to map words to indices in the vocabulary
vocab_dict = {}
for i, word in enumerate(vocab):
    vocab_dict[word] = i

print(len(vocab_dict))
# Initialize BoW matrices for training and testing data


# %%
# reduced_train_tokens = train_tokens

# # Build the BoW representation manually
# # Create a vocabulary by collecting unique words from the training data
# vocab_reduced = set()
# for tokens in reduced_train_tokens:
#     vocab_reduced.update(tokens)

# # Create a dictionary to map words to indices in the vocabulary
# vocab_dict_reduced = {}
# for i, word in enumerate(vocab_reduced):
#     vocab_dict_reduced[word] = i

# print(len(vocab_dict_reduced))

# train_bow = np.zeros((len(reduced_train_tokens), len(vocab_reduced)))

# print("bow train")
# # Convert text to BoW vectors
# for i, tokens in enumerate(reduced_train_tokens):
#     if i % 5000 == 0:
#         print(i)
#     for token in tokens:
#         train_bow[i][vocab_dict_reduced[token]] += 1



# reduced_test_tokens = test_tokens[:5000]
# test_bow = np.zeros((len(reduced_test_tokens), len(vocab_reduced)))

# print("bow test")

# for i, tokens in enumerate(reduced_test_tokens):
#     if i % 1000 == 0:
#         print(i)
#     for token in tokens:
#         if token in vocab_dict_reduced:
#             test_bow[i][vocab_dict_reduced[token]] += 1
#         else:    # if there is an unknown word, add it to the UNK column 
#             test_bow[i][vocab_dict_reduced['UNK']] += 1

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
learning_rate = 0.01
batch_size = 10000

class BoWClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Pass the input through the linear layer
        out = self.linear(x)
        return out
    
# Define the model
input_size = len(vocab)  # Input size is the size of the vocabulary
output_size = 5  # Output size is 5 dimensions

model = BoWClassifier(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(train_tokens), batch_size):
        
        inputs = train_tokens[i:i + batch_size]
        
        inputs = inputs.to(device)
        
        # make bow vector for inputs
        bow = torch.zeros((batch_size, len(vocab)), dtype=torch.float32)
        # print(bow.shape)
        for j in range(batch_size):
            for token in inputs[j]:
                # try :
                bow[j][vocab_dict[token]] += 1
                # except:
                # print(token)
                # bow[j][token] += 1
        
        # convert bow to tensor

        inputs = bow
        
        labels = train_labels[i:i + batch_size]  # Make sure to have train_labels defined
        # Convert labels to LongTensors
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Print the loss for this batch if needed
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i//batch_size+1}], Loss: {loss.item()}')

print('Training finished')


