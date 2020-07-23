print("Loading package 'nlp'")
import nlp
print("Done loading package 'nlp'")

print("Loading package 'transformers'")
import transformers
print("Done loading package 'transformers'")

import matplotlib.pyplot as plt
import numpy as np
import tqdm

def main():
    ############################################################################
    # MODE = "str.split"
    # MAXLEN = 1000

    MODE = "t5-11b"
    MAXLEN = 512
    
    # MODE = "facebook/bart-large"
    # MAXLEN = 1024
    ############################################################################

    print("#" * 80)
    print(f"Mode: {MODE}")
    print(f"Maxlen: {MAXLEN}")
    print("#" * 80)

    # Build the correct tokenize function
    if MODE == "str.split":
        tokenize = str.split
    else:
        def from_tokenizer(tokenizer):
            def tokenize(input_text):
                return tokenizer(input_text)["input_ids"]
            return tokenize
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODE)
        tokenize = from_tokenizer(tokenizer)

    # Load dataset
    print("Loading dataset 'eli5'")
    eli5 = nlp.load_dataset("eli5")
    print("Done loading dataset 'eli5'")

    # Count average length
    lengths = []
    print("Counting...")
    # For each dataset entry
    for entry in tqdm.tqdm(eli5["train_eli5"]):
        # For each possible anser
        for answer in entry["answers"]["text"]:
            tokenized = tokenize(answer)
            length = len(tokenized)
            lengths.append(length)
    
    # Print statistics about the lengths
    lengths = np.array(lengths, dtype=np.int64)
    stddev = np.std(lengths)
    mean = np.mean(lengths)
    median = np.median(lengths)
    delim_long_answer = MAXLEN
    num_long_answers = np.sum(lengths > delim_long_answer)
    print("\n" * 2 + "#" * 80) 
    print("Stats:")
    print("#" * 20)
    print("Mean:", mean)
    print("Median:", median)
    print("standard deviation:", stddev)
    print("min:", np.min(lengths))
    print("max:", np.max(lengths))
    print("count:", len(lengths))
    print(f"qty answers longer than {delim_long_answer}:", num_long_answers)
    print("#" * 80)

    # Plot a histogram with the lengths
    # There are a very small number of extremely long answers that
    # mess up the histogram. We remove these.
    cut_down = lengths[lengths <= mean + stddev * 4]
    plt.hist(cut_down, bins=100, label="length histogram")
    # Add a line at the mean and at the median.
    # There is definitely a cleaner way to do this.
    arbitrary_top_of_line = 20000
    plt.plot([mean, mean], [0, arbitrary_top_of_line], 
             color="blue", label="Mean")
    plt.plot([median, median], [0, arbitrary_top_of_line], 
             color="green", label="Median")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()