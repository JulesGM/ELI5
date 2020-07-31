import importlib
import json
import types
from typing import *

def slow_import(module_name: str) -> types.ModuleType:
    print(f"Loading package '{module_name}'...")
    module_ = importlib.import_module(module_name)
    print(f"Done loading package '{module_name}'.")
    return module_

import absl
from absl import app
from absl import flags

import click
import colored_traceback.auto
import faiss                   
import matplotlib.pyplot as plt
import numpy as np
import torch # This is just to make sure that pytorch is installed
nlp = slow_import("nlp")
transformers = slow_import("transformers")
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_enum("model", "yjernite/bart_eli5", {"yjernite/bart_eli5",}, 
                  "Model to load using `.from_pretrained`.")
flags.DEFINE_integer("log_level", absl.logging.WARNING, "Logging verbosity.")
flags.DEFINE_enum("input_mode", "eli5", {"eli5", "cli"},
                  "")

def print_iterable(iterable: Iterable, extra_return: bool = False) -> None:
    """
    Assumes that there is an end to the iterable, or will run forever.
    """
    print("\n".join([f" - {arg}" for arg in iterable]))
    if extra_return:
        print("")

def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def check_equal(a: Any, b: Any)  -> None:
    check(a == b, f"Expected arguments to be equal, got \n{a} != {b}.")

def check_type(obj: Any, type_: type) -> None:
    check(isinstance(obj, type_), 
          f"Failed isinstance. Expected type `{type_}`, got `{type(obj)}`.")


def main(unused_argv: List[str]) -> None:
    absl.logging.set_verbosity(FLAGS.log_level)

    # Handle unused argvs
    print("Unused argv:")
    print_iterable(unused_argv)
    check(len(unused_argv) <= 1, str(unused_argv))

    # Load the tokenizer    
    tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model)

    # Load the model
    # print(f"Loading model '{FLAGS.model}'...")
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained(FLAGS.model)
    # print(f"Done loading model '{FLAGS.model}'.")

    # Load the snippets from wikipedia
    print(f"Loading dataset 'wiki_dpr' without embeddings...")    
    wiki_dpr = nlp.load_dataset("wiki_dpr", "psgs_w100_no_embeddings")
    # wiki_dpr = nlp.load_dataset("wiki_dpr", "psgs_w100_with_nq_embeddings")
    
    print("#" * 80)
    print(f"Done loading dataset 'wiki_dpr'.")
    print("")
    print("dir(wiki_dpr):", dir(wiki_dpr), "\n")
    print("wiki_dpr['train']:", wiki_dpr["train"], "\n")
    print("dir(wiki_dpr['train']):", dir(wiki_dpr["train"]), "\n")
    print("column names", wiki_dpr["train"].column_names, "\n")
    # print("next iter:", next(iter(wiki_dpr["train"])), "\n")
    # print("getitem:", wiki_dpr["train"][100], "\n")
    print("#" * 80)
        
    if FLAGS.input_mode == "eli5":
        # Load dataset
        print("Loading dataset 'eli5'...")
        eli5 = nlp.load_dataset("eli5")
        print("Done loading dataset 'eli5'.")

        # Attempt to get a prediction
        sample = eli5["train_eli5"][0]
        question = sample["title"]

        check_type(question, str)
        answers = sample["answers"]["text"]
        print("\nQuestion:")
        print(question)

        print("\nLabels:")
        print_iterable(answers)

    elif FLAGS.input_mode == "cli":
        question = input("Question: ")
    
    print("(tokenizing question)")
    tokenized_sample = tokenizer(question, return_tensors="pt")
    print("(predicting answer)")
    sample_predictions = model.generate(**tokenized_sample, 
        do_sample=True,
        max_length=1024, 
        num_return_sequences=10,
        top_k=10)
    print(click.style("\nQuestion:", bold=True), 
          click.style(question, fg="green") + "\n")
    for sample_prediction in sample_predictions:
        print(click.style("decoded output_sequence:", bold=True), 
              tokenizer.decode(sample_prediction).replace("<pad>", ""))
        print("")

if __name__ == "__main__":
    app.run(main)