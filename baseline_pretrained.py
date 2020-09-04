import importlib
import logging
import math
import concurrent.futures as futures
import pathlib
import os
import json
import six
import string
import sys
import textwrap
import time
import types
from typing import *

import absl
# from absl import logging
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

from utils import *
from dpr_utils import *

LOGGER = logging.getLogger(__name__)

QUESTION_INDICES = [1, 11, 111, 1111, 11111, 111111] # Max: 272634
CREATE_MEMMAP_BATCH_SIZE = 512
DEVICE = "cuda:0"
DPR_EMB_DTYPE = np.float32
DPR_K = 5
SAVE_ROOT = pathlib.Path(__file__).parent/"saves"
MAX_LENGTH_CONTEXT = 128
MAX_LENGTH_PREDICTION = 1024
PREDICTION_TOP_K_QUANTITY = 10
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
                    "trivia_qa",
                    {"eli5", "cli", "trivia_qa"},
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
                     False, 
                     "Whether to use the GPU with FAISS.")
flags.DEFINE_enum   ("model", 
                     "yjernite/bart_eli5", 
                     {"yjernite/bart_eli5",}, 
                     "Model to load using `.from_pretrained`.")
flags.DEFINE_integer("log_level", 
                     logging.DEBUG, 
                     "Logging verbosity.")
flags.DEFINE_integer("log_level_hf_nlp", 
                     logging.WARNING, 
                     "Logging verbosity of the very talkative huggingface "
                     "nlp module.")
flags.DEFINE_string ("dpr_faiss_path", 
                    #  str(SAVE_ROOT/"created_embs_PCAR512,IVF262144,SQfp16.faiss"),
                     "/home/mila/g/gagnonju/.cache/huggingface/datasets/wiki_dpr/psgs_w100_with_nq_embeddings/0.0.0/10ec144a48046e48521e2c13993f1ce21c9ab0a702d9ec1eb2b1e604f91eb929/psgs_w100.nq.IVFPQ4096_HNSW32_PQ64-IP-train.faiss",
                     "Save path of the FAISS MIPS index with the DPR "
                     "embeddings.")
flags.DEFINE_string ("dpr_np_memmmap_path", 
                     None, 
                     "Where to save the memory map of the DPR "
                     "embeddings.")
flags.DEFINE_string ("faiss_index_factory", 
                     None,
                     "String describing the FAISS index.")


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
            max_length=MAX_LENGTH_PREDICTION, 
            num_return_sequences=N_GENERATIONS_PER_DISPLAY,
            top_k=PREDICTION_TOP_K_QUANTITY)

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

    # Adjust the root logging level
    logging.basicConfig(level=logging.DEBUG, 
        format="`%(levelname)s` - `%(name)s`: %(message)s")
    # LOGGER.setLevel(logging.DEBUG)
    # This formulation ensures that we are not accidentally creating 
    # a new logger that has no effect, ie, it breaks if "nlp" doesn't
    # already exist.
    # assert "nlp" in logging.Logger.manager.loggerDict
    # logging.getLogger("nlp").setLevel(FLAGS.log_level_hf_nlp)

    def log_loading(message, color="blue"):
        LOGGER.info(click.style(message, fg=color))

    # Argument checks
    if FLAGS.create_dpr_embeddings:
        assert FLAGS.create_np_memmap

    if FLAGS.create_np_memmap:
        assert FLAGS.create_faiss_dpr

    # Load the snippets from wikipedia
    log_loading(f"Loading dataset 'wiki_dpr' with embeddings...")    
    wiki_dpr = nlp.load_dataset("wiki_dpr", "psgs_w100_with_nq_embeddings")
    # if FLAGS.create_np_memmap and FLAGS.create_dpr_embeddings:
    #     print(f"Loading dataset 'wiki_dpr' with embeddings...")    
    #     wiki_dpr = load.load_dataset("wiki_dpr", "psgs_w100_with_nq_embeddings")
    # else:
    #     print(f"Loading dataset 'wiki_dpr' without embeddings...")    
    #     wiki_dpr = load.load_dataset("wiki_dpr", "psgs_w10na0_no_embeddings")
    log_loading(f"Done loading dataset 'wiki_dpr'.", color="green")


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
        log_loading("\nCreating dpr faiss index ...")
        faiss_index_dpr = DenseRetriever.create_index(dpr_np_memmap)
        log_loading("Done creating dpr faiss index.", color="green")

        # Save it.
        log_loading("\nSaving dpr faiss index ...")
        DenseRetriever.save_index(faiss_index_dpr, 
                                  FLAGS.dpr_faiss_path)
        log_loading("Done saving dpr faiss index.", color="green")
        return
    else:
        faiss_index_dpr = DenseRetriever.load_index(
            FLAGS.dpr_faiss_path)

    ############################################################################
    # Load Questions & Search DPR
    ############################################################################
    log_loading("Loading DPRQuestionEncoderTokenizer")
    dpr_tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")
    
    log_loading("Loading DPRQuestionEncoder")
    dpr_question_encoder = transformers.DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")

    log_loading("Loading the ELI5 BART model tokenizer")
    bart_tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model)

    log_loading("Loading the ELI5 BART model")
    bart_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        FLAGS.model)
    log_loading("Done loading models", color="green")

    # Expected format:
    # questions: Iterable[str] with one question per sample
    # answers: Iterable[Optional[List[str]]] with a *list* of at least one answer
    # indices: Iterable[Optional[int]] of the indices 
    if FLAGS.input_mode == "eli5":
        log_loading("Loading ELI5")
        ds = nlp.load_dataset("eli5")["train_eli5"]
        questions = (ds[question_index]["title"].strip()
                     for question_index in QUESTION_INDICES)
        answers = (map(str.strip, ds[question_index]["answers"]["text"])
                   for question_index in QUESTION_INDICES)
        indices = QUESTION_INDICES
        log_loading("Done Loading ELI5", color="green")
    elif FLAGS.input_mode == "trivia_qa":
        log_loading("Loading TriviaQA")
        ds = nlp.load_dataset("trivia_qa", "rc")["train"]
        questions = (ds[question_index]["question"].strip()
                     for question_index in QUESTION_INDICES)
        answers = ([ds[question_index]["answer"]["normalized_value"].strip()]
                   for question_index in QUESTION_INDICES)
        indices = QUESTION_INDICES
        log_loading("Done Loading TriviaQA", color="green")
    elif FLAGS.input_mode == "cli":
        questions = [input("Please write a question: ").strip()]
        answers = [[None]]
        indices = [None]
    else:
        raise RuntimeError(f"Unsupported input_mode: {FLAGS.input_mode}")

    def print_title(text):
        """ This allows us to style all titles from one place.
        A bit too cute, maybe.
        """
        print(click.style(text, bold=True))

    for question, answers, index in zip_safe(questions, answers, indices):
        check_type(question, str)
        number_string = click.style(f"#{index}", fg="blue")

        print_title(f"Question {number_string}:")
        print("")
        print_wrapped(question)
        print("")

        # Print the annotated anwers:
        print_title(f"Labels for {number_string}:")
        print("")        
        print_iterable(answers)
        print("")

        # Retrieve contexts and print them:
        contexts = dpr_faiss_retrieve(faiss_index_dpr=faiss_index_dpr, 
                                      dpr_tokenizer=dpr_tokenizer, 
                                      dpr_question_encoder=dpr_question_encoder 
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


if __name__ == "__main__":
    app.run(main)
