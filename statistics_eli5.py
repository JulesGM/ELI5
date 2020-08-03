"""Computes different statistics about the dataset.
"""
from absl import app
from absl import flags
import nlp
import transformers
import matplotlib.pyplot as plt
import numpy as np
import tqdm

flags.DEFINE_string("mode", "facebook/bart-large", "Which model's tokenize to load from Huggingface. Examples: t5-11b, facebook/bart-large")
flags.DEFINE_integer("maxlen", "1024", "Maximum number of tokens per sample for the encoder, for that model. Examples: 512 for T5, 1024 for BART")

FLAGS = flags.FLAGS

def main(unused_argv):
    if len(unused_argv) > 1:
        raise RuntimeError(f"Got unexpectedly unused elements for argv: {unused_argv}")
        
    ############################################################################
    # Build the correct tokenize function
    ############################################################################
    if FLAGS.mode == "str.split":
        tokenize = str.split
    else:
        def from_tokenizer(tokenizer):
            def tokenize(input_text):
                return tokenizer(input_text)["input_ids"]
            return tokenize
        tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.mode)
        tokenize = from_tokenizer(tokenizer)

    ############################################################################
    # Load dataset
    ############################################################################
    print("Loading dataset 'eli5'")
    eli5 = nlp.load_dataset("eli5")
    print("Done loading dataset 'eli5'")

    ############################################################################
    # Aggregate stats on lengths
    ############################################################################
    lengths = []
    print("Counting...")
    # For each dataset entry
    for entry in tqdm.tqdm(eli5["train_eli5"]):
        # For each possible anser
        for answer in entry["answers"]["text"]:
            tokenized = tokenize(answer)
            length = len(tokenized)
            lengths.append(length)
    
    ############################################################################
    # Print statistics about the lengths
    ############################################################################
    lengths = np.array(lengths, dtype=np.int64)
    stddev = np.std(lengths)
    mean = np.mean(lengths)
    median = np.median(lengths)
    delim_long_answer = FLAGS.maxlen
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

    ############################################################################
    # Plot a histogram with the lengths
    ############################################################################
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
    app.run(main)
