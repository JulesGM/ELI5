""" DPR specific utils
"""
import absl
import concurrent.futures as futures
import faiss
import math
import nlp
import numpy as np
import torch
import time
import transformers
import tqdm

FLAGS = absl.flags.FLAGS
LOGGER = logging.getLogger(__name__)

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
                                 max_length, device):
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
                text, tokenizer, model, MAX_LENGTH_CONTEXT, DEVICE)
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