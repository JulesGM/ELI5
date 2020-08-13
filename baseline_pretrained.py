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
import colored_traceback.always
import faiss
import matplotlib.pyplot as plt
import nlp
import numpy as np
import torch
import transformers
import tqdm

QUESTION_INDICES = [1, 11, 111, 1111, 11111, 111111] # Max: 272634
CREATE_MEMMAP_BATCH_SIZE = 512
DPR_EMB_DTYPE = np.float32
DPR_K = 5
SAVE_ROOT=pathlib.Path(__file__).parent/"saves"
MAX_LENGTH = 128
N_CONTEXTS_PER_DISPLAY = 3
N_GENERATIONS_PER_DISPLAY = 3

FLAGS = flags.FLAGS
# Changes often
flags.DEFINE_boolean("create_faiss_dpr", 
                     False, 
                     "Whether to rebuild DPR's FAISS index")
flags.DEFINE_boolean("create_np_memmap", 
                     False, 
                     "Whether to rebuild DPR's memmap index")
flags.DEFINE_enum   ("input_mode", 
                    #  "eli5", 
                    "cli",
                    {"eli5", "cli"},
                     "What type of input to use for the question.")
flags.DEFINE_boolean("create_dpr_embeddings", 
                     False, 
                     "Whether we generate the embeddings when creating the "
                     "dpr memmap or if we just reuse the ones from the "
                     "Huggingface DB.")
flags.DEFINE_integer("nproc", 
                     len(os.sched_getaffinity(0)),
                     "Number of cpus used to create memmap. Needs to be 1 if "
                     "'create_dpr_embeddings' is true.")
flags.DEFINE_integer("dpr_embedding_depth", 
                     768, 
                     "Depth of the DPR embeddings. If we reuse the embeddings "
                     "from the Huggingface DB, needs to match their depth.")


# Doesn't change quite as often
flags.DEFINE_boolean("faiss_use_gpu", 
                     True, 
                     "Whether to use the GPU with FAISS.")
flags.DEFINE_enum   ("model", 
                     "yjernite/bart_eli5", 
                     {"yjernite/bart_eli5",}, 
                     "Model to load using `.from_pretrained`.")
flags.DEFINE_integer("log_level", 
                     logging.WARNING, 
                     "Logging verbosity.")
flags.DEFINE_string ("dpr_faiss_path", 
                     None,
                     # str(SAVE_ROOT/"hnsw_flat.faiss"), 
                     # str(SAVE_ROOT/"PCAR32,IVF262144_HNSW32,SQfp16.faiss"), 
                     # str(SAVE_ROOT/"PCAR32,IVF65536_HNSW32,SQfp16.faiss"),
                     "Save path of the FAISS MIPS index with the DPR "
                     "embeddings.")
flags.DEFINE_string ("dpr_np_memmmap_path", 
                     None, 
                     "Where to save the memory map of the DPR "
                     "embeddings.")
flags.DEFINE_string ("faiss_index_factory", 
                     None,
                     "String describing the FAISS index.")


################################################################################
# Utilities
################################################################################
def get_terminal_width(default=80):
    try:
        _, columns = os.popen('stty size', 'r').read().split()
    except ValueError as err:
        columns = default
    return int(columns)


def wrap(text, joiner, padding=4):
    wrapped = [line.strip() for line in 
               textwrap.wrap(text, get_terminal_width() - padding)]
    return joiner.join(wrapped)


def print_wrapped(text, first_line="   "):
    print(first_line + wrap(text, "\n   ", 3))


def print_iterable(iterable: Iterable, extra_return: bool = False) -> None:
    """ Assumes that there is an end to the iterable, or will run forever.
    """
    for line in iterable:
            print_wrapped(line, " - ")
            print("")
    
    if extra_return:
        print("")

def print_numbered_iterable(iterable: Iterable, 
                            extra_return: bool = False) -> None:
    """ Assumes that there is an end to the iterable, or will run forever.
    """
    for i, line in enumerate(iterable):
            print_wrapped(line, f"{i}. ")
            print("")
    
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


################################################################################
# Dense Retriever Namespace
################################################################################
class DenseRetriever:
    """Namespace with the functions used for dense retrieval with FAISS.
    """
    def __init__(self):
        raise RuntimeError("Should never be instantiated.")

    @classmethod
    def create_index(cls, embeddings):
        USE_SUBSET = None
        if USE_SUBSET is not None:
            print(f">> CAREFUL. Using subset {USE_SUBSET} / {len(embeddings)},"
                  f" {USE_SUBSET/len(embeddings):0.2%}")
            embeddings = embeddings[:USE_SUBSET]

        DIM = embeddings.shape[1]
        index = faiss.index_factory(DIM, FLAGS.faiss_index_factory,
                                    faiss.METRIC_INNER_PRODUCT)

        # index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
        cls.index_init(index)

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
        return index.search(queries, k)

    @staticmethod
    def index_init(index):
        index.nprobe = 256
        # index.hnsw.efsearch = 128
        # index.hnsw.efConstruction = 200

    @classmethod
    def load_index(cls, path: Union[pathlib.Path, str]):
        print("Loading FAISS index ...")
        index_cpu = faiss.read_index(path)
        cls.index_init(index_cpu)

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
    

def embed_passages_for_retrieval(passages, tokenizer, qa_embedder, 
                                 max_length=128, device="cuda:0"):
    # Ref:
    # https://github.com/huggingface/notebooks/blob/master/longform-qa/lfqa_utils.py

    a_toks = tokenizer.batch_encode_plus(passages, max_length=max_length, 
                                         pad_to_max_length=True)
    a_ids, a_mask = (torch.LongTensor(a_toks["input_ids"]).to(device),
                     torch.LongTensor(a_toks["attention_mask"]).to(device))    
    pooled_output = qa_embedder(a_ids, a_mask)[0].detach()

    return pooled_output.cpu().type(torch.float).numpy()


def create_memmap(path, dataset,  num_embeddings, batch_size, np_index_dtype):
    """Creates the mem-mapped array containing the embeddings for DPR wikipedia.

    Modified from 
    https://github.com/huggingface/notebooks/blob/master/longform-qa/lfqa_utils.py#L566    
    """
    print("Creating memmap.")
    DPR_NUM_EMBEDDINGS = len(dataset)
    print("\t- Creating memmap file...")
    emb_memmap = np.memmap(path,
                           dtype=np_index_dtype, mode="w+", 
                           shape=(num_embeddings, FLAGS.dpr_embedding_depth))
    n_batches = math.ceil(num_embeddings / batch_size)
    avg_duration = ExpRunningAverage(0.1)

    if FLAGS.create_dpr_embeddings:
        print("Loading DPR question encoder tokenizer ...")
        tokenizer = transformers.DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base")
        print("... Done loading DPR tokenizer.")
        print("Loading DPR question encoder model ...")
        model = transformers.DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base").cuda()

    # Prepare the generator for the multiprocessing situation
    def copy_embedding(i):
        start = time.time()
        emb_memmap = np.memmap(FLAGS.dpr_np_memmmap_path,
                               dtype=DPR_EMB_DTYPE, mode="r+", 
                               shape=(DPR_NUM_EMBEDDINGS, 
                               FLAGS.dpr_embedding_depth))
        if FLAGS.create_dpr_embeddings:
            text = [p for p in dataset[i * batch_size : (i + 1) * batch_size][
                    "text"]]
            reps = embed_passages_for_retrieval(
                text, tokenizer, model, MAX_LENGTH)
        else:
            reps = [p for p in dataset[i * batch_size : (i + 1) * batch_size][
                    "embeddings"]]
        emb_memmap[i * batch_size : (i + 1) * batch_size] = reps        
        print(f"{i} / {n_batches} : took {time.time() - start:0f}s - "
              f"writing to '{FLAGS.dpr_np_memmmap_path}'")
        return i
    
    # Ref: https://stackoverflow.com/a/55423170/447599
    print("\t- Starting Pool...")
    duration_full_start = time.time()
    with futures.ThreadPoolExecutor(FLAGS.nproc) as pool:
        for _ in tqdm.tqdm(pool.map(copy_embedding, range(n_batches)), 
                           total=n_batches):
            pass
    duration_full = time.time() - duration_full_start
    tqdm.tqdm.write(f"Took: {tqdm.tqdm.format_interval(duration_full)}")

    print("Done creating mmap.")


def dpr_faiss_retrieve(faiss_index_dpr, 
                       dpr_tokenizer: transformers.DPRQuestionEncoderTokenizer, 
                       dpr_question_encoder: transformers.DPRQuestionEncoder, 
                       question : str):
    check_type(question, str)
    check_type(dpr_tokenizer, transformers.DPRQuestionEncoderTokenizer)
    check_type(dpr_question_encoder, transformers.DPRQuestionEncoder)

    # Tokenize the question for the DPR encoder
    dpr_tokenized = dpr_tokenizer(question, return_tensors="pt")
    
    # Run inference
    dpr_embedding = dpr_question_encoder(**dpr_tokenized)[0]

    # Run search with FAISS
    distances, indices = faiss_index_dpr.search(dpr_embedding.detach().numpy(), 
                                                DPR_K)

    return indices[0]


def bart_predict(bart_tokenizer: transformers.BartTokenizer, 
                bart_model: transformers.modeling_bart.BartForConditionalGeneration,
                question: str, retrieved_contexts: List[str]):

    check_type(bart_tokenizer, transformers.BartTokenizer)
    check_type(bart_model, 
               transformers.modeling_bart.BartForConditionalGeneration)
    check_type(question, str)
    predictions = []

    for i in range(N_CONTEXTS_PER_DISPLAY):    
        # context_string = "<P> " + " <P> ".join(retrieved_contexts) + " <P> "
        context_string = "<P> " + retrieved_contexts[i] + " <P> "
        tokenized_sample = bart_tokenizer(context_string + question,
                                            return_tensors="pt")
        sample_predictions = bart_model.generate(**tokenized_sample, 
            do_sample=True,
            max_length=1024, 
            num_return_sequences=N_GENERATIONS_PER_DISPLAY,
            top_k=10)

        for j in range(N_GENERATIONS_PER_DISPLAY):
            i_txt = click.style(str(i), fg="blue")
            j_txt = click.style(str(j), fg="green")
            predictions.append((f"[ctx {i_txt} / gen {j_txt}]", 
                                sample_predictions[j]))

    return predictions


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

    # Argument checks
    if FLAGS.create_dpr_embeddings:
        assert FLAGS.create_np_memmap

    if FLAGS.create_np_memmap:
        assert FLAGS.create_faiss_dpr

    # Load the snippets from wikipedia
    print(f"Loading dataset 'wiki_dpr' with embeddings...")    
    # wiki_dpr = nlp.load_dataset("wiki_dpr", "psgs_w100_no_embeddings")
    wiki_dpr = nlp.load_dataset("wiki_dpr", "psgs_w100_with_nq_embeddings")
    print(f"Done loading dataset 'wiki_dpr'.")


    ############################################################################
    # Create or load the FAISS index
    ############################################################################
    if not FLAGS.create_dpr_embeddings:
        check_equal(FLAGS.dpr_embedding_depth, len(wiki_dpr['train'
            ][0]["embeddings"]))

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
            if pathlib.Path(FLAGS.dpr_np_memmmap_path).exists():
                raise RuntimeError(
                    f"`{FLAGS.dpr_np_memmmap_path}` already exists. "
                    f"Delete it if you want to create a new DPR Numpy index.")

            create_memmap(FLAGS.dpr_np_memmmap_path, wiki_dpr["train"], 
                          DPR_NUM_EMBEDDINGS, CREATE_MEMMAP_BATCH_SIZE, 
                          DPR_EMB_DTYPE)

        # Open the memory mapped array for reading.
        dpr_np_memmap = np.memmap(FLAGS.dpr_np_memmmap_path,
                                  dtype=DPR_EMB_DTYPE, mode="r", 
                                  shape=(DPR_NUM_EMBEDDINGS, 
                                  FLAGS.dpr_embedding_depth))

        # Actually create the FAISS index.
        print("\nCreating dpr faiss index ...")
        faiss_index_dpr = DenseRetriever.create_index(dpr_np_memmap)
        print("Done creating dpr faiss index.")

        # Save it.
        print("\nSaving dpr faiss index ...")
        DenseRetriever.save_index(faiss_index_dpr, 
                                  FLAGS.dpr_faiss_path)
        print("Done saving dpr faiss index.")
        return
    else:

        faiss_index_dpr = DenseRetriever.load_index(
            FLAGS.dpr_faiss_path)

    ############################################################################
    # Load Questions & Search DPR
    ############################################################################
    dpr_tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")
    
    dpr_question_encoder = transformers.DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")

    bart_tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model)
    bart_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        FLAGS.model)

    eli5 = nlp.load_dataset("eli5")

    if FLAGS.input_mode == "eli5":
        ds = eli5["train_eli5"]
        questions = (ds[question_index]["title"].strip()
                     for question_index in QUESTION_INDICES)
        answers = (map(str.strip, ds[question_index]["answers"]["text"])
                   for question_index in QUESTION_INDICES)
        indices = QUESTION_INDICES
    elif FLAGS.input_mode == "cli":
        questions = [input("Please write a question: ").strip()]
        answers = ["N/A"]
        indices = ["N/A"]
    else:
        raise RuntimeError(f"Unsupported input_mode: {FLAGS.input_mode}")

    def print_title(text):
        """ This allows us to style all titles from one place.
        A bit too cute, maybe.
        """
        print(click.style(text, bold=True))

    for question, answers, index in zip(questions, answers, indices):
        check_type(question, str)
        # number_string = click.style(f"#{question_index}", fg="green")
        number_string = click.style(f"#{index}", fg="blue")

        print_title(f"Question {number_string}:")
        print("")
        print_wrapped(question)
        print("")

        # Print the annotated anwers:
        # print_title("\nLabels:")
        # print_iterable(answers)
        # print("")

        # Retrieve contexts and print them:
        contexts = dpr_faiss_retrieve(faiss_index_dpr=faiss_index_dpr, 
                                      dpr_tokenizer=dpr_tokenizer, 
                                      dpr_question_encoder=dpr_question_encoder, 
                                      question=question)
        context_texts = [wiki_dpr["train"][int(context_index)]["text"] 
                         for context_index in contexts]
        print_title(f"Contexts for question {number_string}:")
        print("")
        print_numbered_iterable(context_texts)
        print("")

        # Print a reminder of the questions
        print_title(f"Question reminder for {number_string}:")
        print("")
        print_wrapped(question)
        print("")

        # Generate answers and print them:
        sample_predictions = bart_predict(bart_tokenizer, bart_model, 
                                          question, context_texts)
        
        formatted_answers = [] 
        for info, sample_prediction in sample_predictions:
            gen_line = bart_tokenizer.decode(sample_prediction)
            gen_line = gen_line.replace("<pad>", "").replace("</s>", "") 
            formatted_answers.append(f"{info}: " + gen_line)
        
        print_title(f"Predictions for question {number_string}:")
        print("")
        print_iterable(formatted_answers)
        print("")
    return 


    ############################################################################
    # BART inference
    ############################################################################
    # Load the tokenizer    


if __name__ == "__main__":
    app.run(main)
