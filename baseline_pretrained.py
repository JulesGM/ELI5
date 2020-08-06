import importlib
import math
import concurrent.futures as futures
import pathlib
import os
import json
import six
import sys
import textwrap
import time
import types
from typing import *

import absl
from absl import logging
from absl import app
from absl import flags
import click
import colored_traceback.auto
import faiss
import matplotlib.pyplot as plt
import nlp
import numpy as np
import torch
import transformers
import tqdm

ELI5_QUESTION_INDEX = 34 # Max: 272634
INDEX_NPROBE = 256
CREATE_MEMMAP_BATCH_SIZE = 512
DPR_EMB_DTYPE = np.float32
DPR_K = 5
SAVE_ROOT=pathlib.Path(__file__).parent/"saves"

FLAGS = flags.FLAGS
# Changes often
flags.DEFINE_string("faiss_index_factory", None,
                    "String describing the FAISS index.")
flags.DEFINE_boolean("create_faiss_dpr", False, 
                     "Whether to rebuild DPR's FAISS index")
flags.DEFINE_boolean("create_np_memmap", False, 
                     "Whether to rebuild DPR's memmap index")
flags.DEFINE_enum   ("input_mode", "eli5", {"eli5", "cli"},
                     "What type of input to use for the question.")

# Doesn't change quite as often
flags.DEFINE_boolean("faiss_use_gpu", True, 
                     "Whether to use the GPU with FAISS.")
flags.DEFINE_enum   ("model", "yjernite/bart_eli5", 
                    {"yjernite/bart_eli5",}, 
                     "Model to load using `.from_pretrained`.")
flags.DEFINE_integer("log_level", logging.WARNING, 
                     "Logging verbosity.")
flags.DEFINE_string ("dpr_faiss_path", 
                     str(SAVE_ROOT/"PCAR32,IVF262144_HNSW32,SQfp16.faiss"), 
                    #  str(SAVE_ROOT/"PCAR32,IVF65536_HNSW32,SQfp16.faiss"),
                     "Save path of the FAISS MIPS index with the DPR "
                     "embeddings.")
flags.DEFINE_string ("dpr_np_memmmap_path", str(SAVE_ROOT/"dpr_np_memmap.dat"), 
                     "Where to save the memory map of the DPR"
                     " embeddings.")


################################################################################
# Utilities
################################################################################
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

class ExpRunningAverage:
    def __init__(self, new_contrib_rate: float):
        self.new_contrib_rate = new_contrib_rate
        self.running_average = None

    def step(self, val):
        if self.running_average is None:
            self.running_average = val
        else:
            self.running_average = (self.new_contrib_rate * val + 
                (1 - self.new_contrib_rate) * self.running_average)
    
        return self.running_average

def get_terminal_width():
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(columns)

def wrap(text, joiner, padding=4):
    wrapped = [line.strip() for line in 
               textwrap.wrap(text, get_terminal_width() - padding)]
    return joiner.join(wrapped)

################################################################################
# Dense Retriever Namespace
################################################################################
class DenseRetriever:
    """Namespace with the functions used for dense retrieval with FAISS.
    """
    def __init__(self):
        raise RuntimeError("Should never be instantiated.")

    @staticmethod
    def create_index(embeddings):
        USE_SUBSET = None
        if USE_SUBSET is not None:
            print(f">> CAREFUL. Using subset {USE_SUBSET} / {len(embeddings)},"
                  f" {USE_SUBSET/len(embeddings):0.2%}")
            embeddings = embeddings[:USE_SUBSET]

        DIM = embeddings.shape[1]
        index = faiss.index_factory(DIM, FLAGS.faiss_index_factory,
                                    faiss.METRIC_INNER_PRODUCT)

        if FLAGS.faiss_use_gpu:
            print("\t- Moving to gpu")
            faiss_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(faiss_res, 0, index)
        
        print("\t- Training the index")
        start = time.time()
        index.train(embeddings)
        print(f"\t- Training took "
              f"{tqdm.tqdm.format_interval(time.time() - start)}")

        print("\t- Adding the embeddings...")
        start = time.time()
        index.add(embeddings)
        print(f"\t- Adding took "
              f"{tqdm.tqdm.format_interval(time.time() - start)}")
        
        
        return index

    @staticmethod
    def search(index, queries, k):
        # One should be careful about assigning to attributes in Python
        assert hasattr(index, "nprobe") 
        index.nprobe = INDEX_NPROBE
        return index.search(queries, k)

    @staticmethod
    def load_index(path: Union[pathlib.Path, str]):
        print("Loading FAISS index ...")
        index_cpu = faiss.read_index(path)
        print("Done loading FAISS index.")
        if FLAGS.faiss_use_gpu:
            print("Moving the DPR FAISS index to the GPU.")
            faiss_res = faiss.StandardGpuResources()
            index_gpu = faiss.index_cpu_to_gpu(faiss_res, 0, index_cpu)
            print("Done moving the DPR FAISS index to the GPU.")
            return index_gpu
        else:
            return index_cpu

    @staticmethod
    def save_index(index, path: Union[pathlib.Path, str]) -> None:
        if FLAGS.faiss_use_gpu:
            print("Moving the index to cpu")
            index = faiss.index_gpu_to_cpu(index)
        print("Saving the index")
        faiss.write_index(index, path)
        print("Done saving the index")
    


def create_memmap(path, dataset, embedding_depth, 
                  num_embeddings, batch_size,
                  np_index_dtype):
    """Creates the mem-mapped array containing the embeddings for DPR wikipedia.

    Modified from 
    https://github.com/huggingface/notebooks/blob/master/longform-qa/lfqa_utils.py#L566    
    """
    print("Creating memmap.")
    DPR_NUM_EMBEDDINGS = len(dataset)
    print("\t- Creating memmap file...")
    emb_memmap = np.memmap(path,
                         dtype=np_index_dtype, mode="w+", 
                         shape=(num_embeddings, embedding_depth))
    n_batches = math.ceil(num_embeddings / batch_size)
    avg_duration = ExpRunningAverage(0.1)

    # Prepare the generator for the multiprocessing situation
    def copy_embedding(i):
        start = time.time()
        emb_memmap = np.memmap(FLAGS.dpr_np_memmmap_path,
                            dtype=DPR_EMB_DTYPE, mode="r+", 
                            shape=(DPR_NUM_EMBEDDINGS, 
                            embedding_depth))
                
        reps = [p for p in dataset[i * batch_size : (i + 1) * batch_size][
            "embeddings"]]
        emb_memmap[i * batch_size : (i + 1) * batch_size] = reps        
        print(f"{i} / {n_batches} : took {time.time() - start:0f}s.")
        return i
    
    # Ref: https://stackoverflow.com/a/55423170/447599
    # nproc = len(os.sched_getaffinity(0))
    nproc = 64
    print("\t- Starting Pool...")
    duration_full_start = time.time()
    with futures.ThreadPoolExecutor(nproc) as pool:
        for _ in tqdm.tqdm(pool.map(copy_embedding, range(n_batches)), 
                           total=n_batches):
            pass
    duration_full = time.time() - duration_full_start
    tqdm.tqdm.write(f"Took: {tqdm.tqdm.format_interval(duration_full)}")

    print("Done creating mmap.")
    

################################################################################
# Main
################################################################################
def main(unused_argv: List[str]) -> None:
    # Handle unused argvs
    print("Unused argv:")
    print_iterable(unused_argv)
    check(len(unused_argv) <= 1, str(unused_argv))

    # Adjust the logging level
    logging.set_verbosity(FLAGS.log_level)

    # Load the snippets from wikipedia
    print(f"Loading dataset 'wiki_dpr' with embeddings...")    
    # wiki_dpr = nlp.load_dataset("wiki_dpr", "psgs_w100_no_embeddings")
    wiki_dpr = nlp.load_dataset("wiki_dpr", "psgs_w100_with_nq_embeddings")
    print(f"Done loading dataset 'wiki_dpr'.")


    ############################################################################
    # Create or load the FAISS index
    ############################################################################
    DPR_EMBEDDING_DEPTH = len(wiki_dpr['train'][0]["embeddings"])
    DPR_NUM_EMBEDDINGS = len(wiki_dpr['train'])
    if FLAGS.create_faiss_dpr:
        # We will create the FAISS index.
        # But first, we have to prepare the context embeddings by
        # transferring them from the huggingface dataset object to
        # a Numpy memory mapped array. 

        if pathlib.Path(FLAGS.dpr_faiss_path).exists():
            raise RuntimeError(
                f"`{FLAGS.dpr_faiss_path}` already exists. "
                f"Delete it if you want to create a new FAISS DPR index.")

        if FLAGS.create_np_memmap:
            # Create the memory mapped array.
            if pathlib.Path(FLAGS.dpr_np_memmmap_patindex_toh).exists():
                raise RuntimeError(
                    f"`{FLAGS.dpr_np_memmmap_path}` already exists. "
                    f"Delete it if you want to create a new DPR Numpy index.")

            create_memmap(FLAGS.dpr_np_memmmap_path, wiki_dpr["train"], 
                          DPR_EMBEDDING_DEPTH, DPR_NUM_EMBEDDINGS,
                          CREATE_MEMMAP_BATCH_SIZE, DPR_EMB_DTYPE)

        # Open the memory mapped array for reading.
        dpr_np_memmap = np.memmap(FLAGS.dpr_np_memmmap_path,
                                  dtype=DPR_EMB_DTYPE, mode="r", 
                                  shape=(DPR_NUM_EMBEDDINGS, 
                                  DPR_EMBEDDING_DEPTH))

        # Actually create the FAISS index.
        print("\nCreating dpr faiss index ...")
        faiss_index_dpr = DenseRetriever.create_index(dpr_np_memmap)
        print("Done creating dpr faiss index.")

        # Save it.
        print("\nSaving dpr faiss index ...")
        DenseRetriever.save_index(faiss_index_dpr, 
                                  FLAGS.dpr_faiss_path)
        print("Done saving dpr faiss index.")
    else:

        faiss_index_dpr = DenseRetriever.load_index(
            FLAGS.dpr_faiss_path)

    ############################################################################
    # Load the questions dataset, or read inputs from the user.
    ############################################################################
    if FLAGS.input_mode == "eli5":
        # Load dataset
        print("Loading dataset 'eli5' ...")
        eli5 = nlp.load_dataset("eli5")
        print("Done loading dataset 'eli5'.")

        # Attempt to get a prediction
        sample = eli5["train_eli5"][ELI5_QUESTION_INDEX]
        question = sample["title"]

        check_type(question, str)
        answers = sample["answers"]["text"]
        print("\nQuestion:")
        print("   " + question)

        print("\nLabels:")
        for answer in answers:
            print(" - " + wrap(answer, "\n   ", 3) + "\n")
        print("")

    elif FLAGS.input_mode == "cli":
        question = input("Question: ")


    ############################################################################
    # DPR Search
    ############################################################################
    # print("Loading DPR tokenizer ...")
    dpr_tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")
    # print("... Done loading DPR tokenizer.")

    # print("Loading DPR question encoder model ...")
    dpr_model = transformers.DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")
    # bert_base = transformers.BertModel.from_pretrained("bert-base-uncased")
    # dpr_model.question_encoder.bert_model.embeddings.position_ids = (
    #     bert_base.embeddings.position_ids)
    # print("... Done loading DPR question encoder model.")

    # Tokenize the question for the DPR encoder
    dpr_tokenized = dpr_tokenizer(question, return_tensors="pt")
    
    # Run the DPR encoder to get the embedding key to search FAISS with
    # print("Predicting retrieval embedding ...")
    # Ref: 
    # https://huggingface.co/transformers/master/model_doc/dpr.html#dprquestionencoder
    dpr_embedding = dpr_model(**dpr_tokenized)[0]

    # Do the FAISS search
    # print("Done predicting retrieval embedding.")
    distances, indices = faiss_index_dpr.search(dpr_embedding.detach().numpy(), DPR_K)
    # print("Indices:", indices.shape)
    print("Contexts:\n")
    for index in indices[0]:
        print(" - " + wrap(wiki_dpr["train"][int(index)]["text"], "\n   ", 3) 
              + "\n")

    return 


    ############################################################################
    # BART inference
    ############################################################################
    # Load the tokenizer    
    bart_tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model)

    print("\nTokenizing question..")
    tokenized_sample = bart_tokenizer(question, return_tensors="pt")


    print(f"\nLoading model '{FLAGS.model}'...")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        FLAGS.model)
    print(f"Done loading model '{FLAGS.model}'.")

    print("\nPredicting answer")
    sample_predictions = model.generate(**tokenized_sample, 
        do_sample=True,
        max_length=1024, 
        num_return_sequences=10,
        top_k=10)

    # Print the results:
    print(click.style("\nQuestion:", bold=True), 
          click.style(question, fg="green") + "\n")
    for sample_prediction in sample_predictions:
        print(click.style("decoded output_sequence:", bold=True), 
              tokenizer.decode(sample_prediction).replace("<pad>", ""))
        print("")

if __name__ == "__main__":
    app.run(main)