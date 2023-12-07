from datasets import load_dataset
import tiktoken

dataset = load_dataset("financial_phrasebank", 'sentences_allagree')
enc = tiktoken.get_encoding("cl100k_base")

def write_data(dataset, filename):
    """
    Write the sentences to a file
    """
    # concatenate all the sentences into one list
    sentences = []
    for split in dataset.keys():
        sentences += dataset[split]['sentence']

    # Clean the sentences

    # concatenate all the sentences into one string
    data = "\n".join(sentences)

    # write the data to a file
    with open(filename, "w") as f:
        f.write(data)

def tokenize(path: str):
    """
    Tokenize the data
    """
    # read the data from the file
    with open(path, "r") as f:
        data = f.read()

    # tokenize the data
    tokenized_data = enc.encode(data)
    print(f"Tokenized {len(tokenized_data)} tokens")

if __name__ == "__main__":
    tokenize("finance.txt")

