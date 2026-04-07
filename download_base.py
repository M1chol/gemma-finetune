from huggingface_hub import login, snapshot_download

login()

local_dir = "models/functiongemma-270m-it"
snapshot_download(
    repo_id="google/functiongemma-270m-it",
    local_dir=local_dir,
)
