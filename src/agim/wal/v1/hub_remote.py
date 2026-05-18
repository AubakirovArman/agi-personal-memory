from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .nn import WALParameter
from .hub_card import WALModelCard
from .hub_state import extract_wal_state_dict


def push_wal_model(
    model: nn.Module,
    repo_id: str,
    card: WALModelCard,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload WAL-encoded model",
) -> str:
    """Push a WAL-encoded model to the Hugging Face Hub.
    
    Args:
        model: PyTorch model with WAL layers
        repo_id: HF Hub repo ID (e.g., "username/model-name")
        card: Model metadata card
        token: HF API token
        private: Whether repo should be private
        commit_message: Git commit message
    
    Returns:
        URL of the uploaded model
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    api = HfApi(token=token)
    
    # Create repo if needed
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
    except Exception:
        pass
    
    # Extract WAL state
    wal_state = extract_wal_state_dict(model)
    
    # Prepare upload directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save WAL blobs
        wal_dir = tmpdir / "wal_weights"
        wal_dir.mkdir()
        for name, blob in wal_state["wal_blobs"].items():
            safe_name = name.replace(".", "_") + ".wal"
            (wal_dir / safe_name).write_bytes(blob)
        
        # Save biases
        if wal_state["biases"]:
            bias_path = tmpdir / "biases.safetensors"
            from safetensors.torch import save_file
            save_file(wal_state["biases"], str(bias_path))
        
        # Save non-WAL params
        if wal_state["non_wal"]:
            non_wal_path = tmpdir / "non_wal.safetensors"
            from safetensors.torch import save_file
            save_file(wal_state["non_wal"], str(non_wal_path))
        
        # Save metadata
        metadata = {
            "card": card.to_dict(),
            "wal_layers": wal_state["wal_layers"],
        }
        (tmpdir / "wal_config.json").write_text(json.dumps(metadata, indent=2))
        
        # Upload
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            commit_message=commit_message,
        )
    
    return f"https://huggingface.co/{repo_id}"


def pull_wal_model(
    repo_id: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Pull a WAL-encoded model from the Hugging Face Hub.
    
    Args:
        repo_id: HF Hub repo ID
        token: HF API token
        cache_dir: Local cache directory
    
    Returns:
        Dict with:
        - 'wal_params': reconstructed WALParameters
        - 'biases': bias tensors
        - 'non_wal': non-WAL parameters
        - 'card': WALModelCard
        - 'wal_layers': layer metadata
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    # Download files
    config_path = hf_hub_download(
        repo_id=repo_id, filename="wal_config.json",
        token=token, cache_dir=cache_dir,
    )
    config = json.loads(Path(config_path).read_text())
    card = WALModelCard.from_dict(config["card"])
    wal_layers = config.get("wal_layers", {})
    
    result = {
        "card": card,
        "wal_layers": wal_layers,
        "wal_params": {},
        "biases": {},
        "non_wal": {},
    }
    
    # Download and deserialize WAL blobs
    from .format import deserialize_wal_v1
    
    wal_dir = Path(config_path).parent / "wal_weights"
    if wal_dir.exists():
        for wal_file in wal_dir.glob("*.wal"):
            blob = wal_file.read_bytes()
            name = wal_file.stem.replace("_", ".")
            prog, atom_table, coeffs, meta = deserialize_wal_v1(blob)
            layer_name = name.replace(".wal_weight", "")
            layer_info = wal_layers.get(layer_name, {})
            shape = tuple(layer_info.get("shape", prog.shape))
            dtype = getattr(torch, layer_info.get("dtype", "float32"))
            
            wal_param = WALParameter(
                prog=prog,
                atom_table=atom_table,
                coeffs=coeffs,
                shape=shape,
                dtype=dtype,
            )
            result["wal_params"][name] = wal_param
    
    # Download biases
    try:
        bias_path = hf_hub_download(
            repo_id=repo_id, filename="biases.safetensors",
            token=token, cache_dir=cache_dir,
        )
        from safetensors.torch import load_file
        result["biases"] = load_file(bias_path)
    except Exception:
        pass
    
    # Download non-WAL params
    try:
        non_wal_path = hf_hub_download(
            repo_id=repo_id, filename="non_wal.safetensors",
            token=token, cache_dir=cache_dir,
        )
        from safetensors.torch import load_file
        result["non_wal"] = load_file(non_wal_path)
    except Exception:
        pass
    
    return result
