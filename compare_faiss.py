
import dpr_utils
import utils

import absl
import click
import faiss
import nlp
import numpy as np
import transformers

LOGGER = logging.getLogger(__name__)

SAVE_ROOT = pathlib.Path(__file__).parent/"saves"

QUESTION_INDICES = [1, 11, 111]

flags.DEFINE_string("url0",
    str(SAVE_ROOT/"created_embs_PCAR512,IVF262144,SQfp16.faiss"),
    "")
flags.DEFINE_string("url1",
      "/home/mila/g/gagnonju/.cache/huggingface/datasets/wiki_dpr/psgs_w100_with_nq_embeddings/0.0.0/10ec144a48046e48521e2c13993f1ce21c9ab0a702d9ec1eb2b1e604f91eb929/psgs_w100.nq.IVFPQ4096_HNSW32_PQ64-IP-train.faiss",
    "")
FLAGS = absl.flags.FLAGS


def main(_):
    indices = []
    logger = utils.LoggingFormatter(LOGGER)

    for index_path in [FLAGS.url0, FLAGS.url1]:
        logger.log_loading(f"Loading index with path: {index_path}")
        indices.append(faiss.read_index(index_path))
        logger.log_loading(f"Done loading that one.", color="green")

    logger.log_loading("Loading the dataset:")
    ds = nlp.load_dataset("eli5")["train_eli5"]
    logger.log_loading(f"Done loading that one.", color="green")

    questions = (ds[question_index]["title"].strip()
                 for question_index in QUESTION_INDICES)

    logger.log_loading("Loading DPRQuestionEncoderTokenizer")
    dpr_tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")
    logger.log_loading(f"Done loading that one.", color="green")

    logger.log_loading("Loading DPRQuestionEncoder")
    dpr_question_encoder = transformers.DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base")
    logger.log_loading(f"Done loading that one.", color="green")

    for question in questions:
        contexts = (dpr_utils.dpr_faiss_retrieve(
                        faiss_index_dpr=indices[0], 
                        dpr_tokenizer=dpr_tokenizer, 
                        dpr_question_encoder=dpr_question_encoder 
                        question=question),
                    dpr_utils.dpr_faiss_retrieve(faiss_index_dpr=indices[1], 
                        dpr_tokenizer=dpr_tokenizer, 
                        dpr_question_encoder=dpr_question_encoder 
                        question=question)
                    )




if __name__ == "__main__":
    abs.app.run(main)