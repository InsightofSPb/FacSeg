import logging
import numpy as np
import struct
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _is_tensor_state_dict(state) -> bool:
    """Return ``True`` if *state* looks like a torch state_dict."""
    if not isinstance(state, (dict, OrderedDict)):
        return False
    if not state:
        return False
    return all(isinstance(value, torch.Tensor) for value in state.values())


def _strip_module_prefix(state_dict):
    """Remove a leading ``module.`` prefix introduced by ``DataParallel``."""
    prefix = "module."
    keys = list(state_dict.keys())
    if keys and all(key.startswith(prefix) for key in keys):
        return OrderedDict((key[len(prefix):], value) for key, value in state_dict.items()), True
    return state_dict, False


def load_checkpoint_state(path: str, device) -> Tuple[OrderedDict, str, bool, Optional[dict]]:
    """Load a checkpoint and return a clean ``state_dict``.

    The helper understands both bare ``state_dict`` files and the wrapped
    payloads produced by :func:`torch.save` in the LPOSS training pipeline,
    where the actual weights live under a ``state_dict`` key.  It also strips
    the ``module.`` prefix that appears when checkpoints are saved from a
    ``DataParallel`` model.

    Returns
    -------
    state_dict:
        The extracted state dictionary, ready to be passed to
        :meth:`torch.nn.Module.load_state_dict`.
    source_key:
        ``"root"`` when the file already contained a bare ``state_dict`` or
        the dictionary key that held the state otherwise.
    stripped_module_prefix:
        ``True`` if a leading ``module.`` prefix was removed from all keys.
    """

    payload = torch.load(path, map_location=device)
    state_dict = None
    source_key = "root"
    metadata = None

    if _is_tensor_state_dict(payload):
        state_dict = payload
    elif isinstance(payload, (dict, OrderedDict)):
        metadata = payload.get("metadata")
        for candidate in ("state_dict", "model_state", "model_state_dict", "model"):
            maybe_state = payload.get(candidate)
            if _is_tensor_state_dict(maybe_state):
                state_dict = maybe_state
                source_key = candidate
                break

    if state_dict is None:
        raise ValueError(
            f"Checkpoint '{path}' does not contain a recognisable state_dict."
        )

    state_dict, stripped = _strip_module_prefix(state_dict)
    return state_dict, source_key, stripped, metadata

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S        # 也是生成X和Y
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

def reorder_data(data, batchsize, iter_num):
    arr = list()
    for i in range(batchsize):
        for j in range(iter_num):
            arr.append(batchsize * j + i)
    return np.array(data[arr])

def var_int_encode(byte_str_len, f):  # 这段代码是用于对整数进行变长编码的函数。它的目的是将一个整数按照一定规则编码成一个字节序列，并将编码后的字节写入文件对象 f 中。
    while True:
        this_byte = byte_str_len & 127
        byte_str_len >>= 7
        if byte_str_len == 0:
            f.write(struct.pack('B', this_byte))
            break
        f.write(struct.pack('B', this_byte | 128))
        byte_str_len -= 1

def split_data(file, prefix, n, tempfile):
    with open(file, 'rb') as f:  # 一次一个byte = 8bit
        series = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()
    vals = list(set(series))
    vals.sort()
    char2id_dict = {str(c): i for (i, c) in enumerate(vals)}
    id2char_dict = {str(i): c for (i, c) in enumerate(vals)}
    params = dict()
    segment_length = len(series) // n
    for i in range(n):
        start_index = i * segment_length
        # 最后一段可能包括剩余的部分
        end_index = start_index + segment_length if i < n - 1 else len(series)
        segment = series[start_index:end_index]
        fout = open(tempfile + '/' + prefix + '.' + str(i), 'wb')
        fout.write(bytearray(segment))
        fout.close()
        params[prefix + '.' + str(i)] = len(segment)
    params['char2id_dict'] = char2id_dict
    params['id2char_dict'] = id2char_dict
    with open(prefix + '.params', 'w') as f:
        f.write(str(params))
    f.close()

def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
            break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len


def infer_msdzip_model_config(state_dict: OrderedDict) -> Dict[str, int]:
    """Infer ``MixedModel`` hyper-parameters from a checkpoint state dict."""

    config: Dict[str, int] = {}

    embedding = state_dict.get("embedding.weight")
    if embedding is not None:
        config["vocab_size"] = int(embedding.shape[0])
        config["vocab_dim"] = int(embedding.shape[1])

    linear = state_dict.get("lin.weight")
    if linear is not None:
        config.setdefault("vocab_size", int(linear.shape[0]))
        config["hidden_dim"] = int(linear.shape[1])

    w1 = state_dict.get("W1")
    if w1 is not None:
        config["layers"] = int(w1.numel())

    for key, tensor in state_dict.items():
        if key.endswith("sgu.spatial_proj.weight"):
            config["timesteps"] = int(tensor.shape[0])
            break

    for key, tensor in state_dict.items():
        if key.endswith("V_map.U"):
            config["batchsize"] = int(tensor.shape[0])
            if "vocab_dim" not in config and tensor.ndim >= 2:
                config["vocab_dim"] = int(tensor.shape[1])
            break

    for key, tensor in state_dict.items():
        if key.endswith("U_map.weight"):
            hidden_dim = int(tensor.shape[1])
            config.setdefault("hidden_dim", hidden_dim)
            ffn_dim = int(tensor.shape[0])
            if ffn_dim % 2 == 0:
                ffn_dim //= 2
            config["ffn_dim"] = ffn_dim
            break

    return config


def apply_checkpoint_overrides(args: Any, config: Dict[str, int], *, logger: Optional[logging.Logger] = None) -> Dict[str, Tuple[Any, Any]]:
    """Update argument namespace so it matches checkpoint hyper-parameters."""

    logger = logger or logging.getLogger(__name__)
    overrides: Dict[str, Tuple[Any, Any]] = {}
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        value = int(value)
        current = getattr(args, key)
        if current != value:
            overrides[key] = (current, value)
            logger.info("Checkpoint override: %s=%s (was %s)", key, value, current)
        setattr(args, key, value)
    return overrides


def prepare_state_from_weights(args: Any, device, *, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """Load checkpoint weights and align CLI arguments with the stored model."""

    if not getattr(args, "weights", None):
        return None

    logger = logger or logging.getLogger(__name__)
    state_dict, source_key, stripped, metadata = load_checkpoint_state(args.weights, device)
    config = infer_msdzip_model_config(state_dict)
    overrides = apply_checkpoint_overrides(args, config, logger=logger)
    logger.info("Loaded checkpoint from %s (source key: %s)", args.weights, source_key)
    if config:
        logger.info("Checkpoint-implied model configuration: %s", config)
    return {
        "state_dict": state_dict,
        "source_key": source_key,
        "stripped": stripped,
        "metadata": metadata,
        "config": config,
        "overrides": overrides,
    }
