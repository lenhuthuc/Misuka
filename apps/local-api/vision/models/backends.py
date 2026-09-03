"""Thread-capped inference sessions, shared by the detector and the embedder.

One small abstraction over two runtimes because the choice is per-machine, not
per-model: this box has no CUDA, so the only question is whether a given model
runs faster on the Iris Xe through OpenVINO or on the P-cores through ONNX
Runtime, and that is answered by `scripts/bench.py`, not in code.

Both backends are capped to the same intra-op thread budget. Uncapped, ONNX
Runtime grabs all 16 hardware threads and the chat/TTS path -- which shares
this process -- loses its cores mid-turn.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class InferenceBackend(Protocol):
    """A loaded model reduced to what this layer needs: name in, arrays out."""

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...

    @property
    def input_names(self) -> list[str]: ...


def apply_thread_env(num_threads: int) -> None:
    """Cap OpenMP before any runtime is imported.

    Both ONNX Runtime and OpenVINO read these at library-load time, so setting
    them after the first session is built has no effect -- hence a module-level
    helper the model wrappers call in their constructors, before importing.
    """
    for var in ("OMP_NUM_THREADS", "INFERENCE_NUM_THREADS", "OPENVINO_NUM_THREADS"):
        os.environ.setdefault(var, str(num_threads))


class OnnxBackend:
    """ONNX Runtime, CPU execution provider."""

    def __init__(self, model_path: Path, num_threads: int = 8) -> None:
        apply_thread_env(num_threads)
        import onnxruntime as ort

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run `python scripts/export_models.py` first."
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        # The graphs here are single-branch; a second pool would only contend.
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        logger.info(
            "vision: loaded %s via onnxruntime (%d threads)", model_path.name, num_threads
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._session.run(None, inputs)

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)


class OpenVINOBackend:
    """OpenVINO IR, iGPU with an automatic CPU fallback.

    The fallback is not defensive noise: the Iris Xe driver refuses to compile
    some quantised graphs, and on this machine the failure surfaces only at
    compile time, well after the model file itself has validated.
    """

    def __init__(self, model_path: Path, num_threads: int = 8, device: str = "GPU") -> None:
        apply_thread_env(num_threads)
        import openvino as ov

        if not model_path.exists():
            raise FileNotFoundError(
                f"OpenVINO IR not found at {model_path}. "
                "Run `python scripts/export_models.py --runtime openvino` first."
            )

        core = ov.Core()
        model = core.read_model(str(model_path))
        available = core.available_devices

        target = device if device in available else "CPU"
        if target != device:
            logger.warning(
                "vision: OpenVINO device %r unavailable (have %s), using CPU",
                device, available,
            )
        try:
            self._compiled = core.compile_model(
                model, target, {"INFERENCE_NUM_THREADS": str(num_threads)}
            )
        except Exception as exc:  # driver refused the graph -- see docstring
            if target == "CPU":
                raise
            logger.warning("vision: OpenVINO %s compile failed (%s), falling back to CPU", target, exc)
            target = "CPU"
            self._compiled = core.compile_model(
                model, "CPU", {"INFERENCE_NUM_THREADS": str(num_threads)}
            )

        self._request = self._compiled.create_infer_request()
        self._input_names = [next(iter(inp.names), f"input_{i}") for i, inp in enumerate(self._compiled.inputs)]
        self._device = target
        logger.info(
            "vision: loaded %s via openvino on %s (%d threads)",
            model_path.name, target, num_threads,
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        # Positional feed: OpenVINO matches by port, and the exported IR keeps
        # the ONNX input order.
        ordered = [inputs[name] for name in self._input_names if name in inputs]
        if len(ordered) != len(self._input_names):
            ordered = list(inputs.values())
        self._request.infer(ordered)
        return [self._request.get_output_tensor(i).data for i in range(len(self._compiled.outputs))]

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def device(self) -> str:
        return self._device


def load_backend(
    *,
    runtime: str,
    onnx_path: Path,
    openvino_path: Path | None = None,
    num_threads: int = 8,
    openvino_device: str = "GPU",
) -> InferenceBackend:
    """Pick a backend, degrading to ONNX rather than failing the ingest.

    A missing OpenVINO install or IR directory is a deployment detail, not a
    reason for the whole vision layer to be unavailable.
    """
    if runtime == "openvino" and openvino_path is not None:
        try:
            return OpenVINOBackend(openvino_path, num_threads=num_threads, device=openvino_device)
        except Exception as exc:
            logger.warning("vision: OpenVINO backend unavailable (%s), using onnxruntime", exc)
    return OnnxBackend(onnx_path, num_threads=num_threads)
