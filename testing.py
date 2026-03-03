vocab=["a","s","d","f","g","h"]
encode = lambda s: [vocab.index(c) for c in s]
print(encode("dsa"))