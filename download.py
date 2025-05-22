from huggingface_hub import snapshot_download, hf_hub_url, list_repo_files
import os

# Define the target folder
# target_folder = "./my_models/inf-retriever-v1-1.5b"
# repo_id = "infly/inf-retriever-v1-1.5b"

target_folder = "./reranker_models/reranker-msmarco-ModernBERT-base-lambdaloss"
repo_id = "tomaarsen/reranker-msmarco-ModernBERT-base-lambdaloss"

# List all files in the repo
all_files = list_repo_files(repo_id)

# Filter to only *.json and *.safetensors in the root directory
allowed_files = [f for f in all_files if "/" not in f and (f.endswith(".json") or f.endswith(".safetensors") or (f.endswith(".py")))]


# Download only those files
snapshot_download(
    repo_id=repo_id,
    local_dir=target_folder,
    local_dir_use_symlinks=False,
    allow_patterns=["*.py", "*.json", "*.safetensors"],
)