#First we download our dataset

#now retrieve it to process
with open('emails.txt', 'r') as file:
    data = file.read()

#let's check a sample
print(f"First hundred characters are: {data[:100]}")

#Find our vocabulary in terms of characters
vocab = sorted(list(set(data)))
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}, list of characters: {vocab}")

