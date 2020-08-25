## How to run: Inference
1. Make sure the saved FAISS index matches the one in `baseline_pretrained.py`'s arguments.
2. (Go on a compute node)
3. Run `source load_env.sh`
4. Run `python baseline_pretrained.py`

## How to run: Setup
- Modify the `launch_jobs.sh` script to generate the Numpy `.dat` file with the DPR BERT embeddings for the Wikipedia context, then run it.
- Modify the `launch_jobs.sh` script to generate the FAISS index, then run it.
For `launch_jobs.sh`, you can use `-i` to run the job interactively in the current computation node.
