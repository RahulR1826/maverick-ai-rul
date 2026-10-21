"""
research/comm.py
----------------
Communication accounting for FedAvg.

What is MEASURED here: the size in bytes of the model update exactly as the
Flower framework serialises it (``flwr.common.ndarray_to_bytes`` writes every
state-dict tensor with ``np.save(..., allow_pickle=False)``); this function
reproduces that serialisation byte-for-byte (verified in tests when flwr is
importable).  It is the application-level payload only.

What is NOT measured: gRPC/protobuf framing, TLS, TCP/IP headers, retransmissions,
client-selection/config messages, metric messages.  No real network was used --
clients are simulated in one process -- so all traffic figures are
"serialized-payload estimates", not measured network traffic.

Protocol assumptions (full participation, synchronous FedAvg):
  * each round the server sends the global model to each of the K selected
    clients individually (K unicast downloads, no broadcast/multicast savings);
  * each selected client uploads one full fp32 model (no compression);
  * the initial global model download in round 1 is counted; the final model
    distribution after the last round is not.
"""

import io
from typing import Dict, List

import numpy as np


def state_arrays(model: "torch.nn.Module") -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def serialized_payload_bytes(arrays: List[np.ndarray]) -> int:
    total = 0
    for a in arrays:
        buf = io.BytesIO()
        np.save(buf, a, allow_pickle=False)
        total += len(buf.getvalue())
    return total


def payload_info(model: "torch.nn.Module") -> Dict[str, float]:
    arrays = state_arrays(model)
    n_params = int(sum(a.size for a in arrays))
    raw = int(sum(a.nbytes for a in arrays))
    ser = serialized_payload_bytes(arrays)
    return {
        "n_tensors": len(arrays), "n_parameters": n_params,
        "dtype": str(arrays[0].dtype),
        "analytic_bytes_params_x4": n_params * 4,
        "raw_tensor_bytes": raw,
        "serialized_payload_bytes": ser,
        "serialization_overhead_bytes": ser - raw,
    }


def fedavg_traffic(payload_bytes: int, clients_per_round: List[int]) -> Dict:
    per_round = []
    cum_up = cum_down = 0
    for r, k in enumerate(clients_per_round, start=1):
        up, down = k * payload_bytes, k * payload_bytes
        cum_up += up
        cum_down += down
        per_round.append({"round": r, "participating_clients": k, "upload_bytes": up,
                          "download_bytes": down, "cum_upload_bytes": cum_up,
                          "cum_download_bytes": cum_down, "cum_total_bytes": cum_up + cum_down})
    return {"per_round": per_round, "total_upload_bytes": cum_up, "total_download_bytes": cum_down,
            "total_bytes": cum_up + cum_down,
            "label": "serialized-payload estimate (application payload only; no protocol overhead; simulated, not measured on a network)"}
