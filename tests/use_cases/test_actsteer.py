# tests/controls/test_actsteer.py
from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM, register_plugins
from vllm_hook_plugins.workers import steer_activation_worker
from tests.conftest import ensure_config_for_model

TEST_MODELS = [
    "facebook/opt-125m",
    "gpt2",
    "Qwen/Qwen2-1.5B-Instruct",
]


@pytest.mark.parametrize("model_id", TEST_MODELS)
def test_activation_steer(cache_dir, project_root, model_id):
    register_plugins()

    cfg = ensure_config_for_model(project_root, "activation_steer", model_id)

    llm = HookLLM(
        model=model_id,
        worker_name="steer_hook_act",
        analyzer_name=None,
        config_file=str(cfg),
        download_dir=str(cache_dir),
        gpu_memory_utilization=0.5,
        dtype=torch.float16,
        enable_hook=True,
    )

    prompt = "This is for testing only."

    _ = llm.generate(prompt, max_tokens=10, temperature=0.0, use_hook=True)


class _OneLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.decoder = torch.nn.Module()
        self.model.decoder.layers = torch.nn.ModuleList([torch.nn.Identity()])


def _request(extra_args, output_token_ids=()):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=extra_args),
        output_token_ids=list(output_token_ids),
    )


def _top_logprobs(output):
    return tuple(
        tuple(sorted((token_id, logprob.logprob) for token_id, logprob in step.items()))
        for step in output.outputs[0].logprobs
    )


def test_activation_steer_worker_lifecycle_cpu(tmp_path, monkeypatch):
    vector_path = tmp_path / "steering.pt"
    vector = torch.tensor([1.0, -2.0, 3.0, -4.0])
    torch.save({"dir": vector, "avg_proj": torch.tensor(0.0)}, vector_path)

    model = _OneLayerModel()
    on = {
        "steer": {
            "method": "add_vector",
            "coefficient": 2.0,
            "optimal_layer": 0,
            "vector_path": str(vector_path),
            "apply_at_all_positions": True,
        }
    }
    zero = {"steer": {**on["steer"], "coefficient": 0.0}}
    requests = {"on": _request(on), "off": _request(None)}
    runner = SimpleNamespace(
        model=model,
        input_batch=SimpleNamespace(req_ids=["on", "off"]),
        requests=requests,
    )
    worker = steer_activation_worker.SteerHookActWorker()
    worker.model_runner = runner
    worker._install_hooks()
    assert len(worker._hooks) == 1

    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 3, 5]))
    monkeypatch.setattr(
        steer_activation_worker,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )

    prefill = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    expected_prefill = prefill.clone()
    expected_prefill[2] += 2 * vector
    observed_prefill = model.model.decoder.layers[0](prefill.clone())
    assert torch.equal(observed_prefill, expected_prefill)
    assert torch.equal(observed_prefill[:2], prefill[:2])
    assert torch.equal(observed_prefill[3:], prefill[3:])

    requests["on"].output_token_ids = [7]
    requests["off"].output_token_ids = [8]
    metadata.query_start_loc = torch.tensor([0, 1, 2])
    decode = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected_decode = decode.clone()
    expected_decode[0] += 2 * vector
    observed_decode = model.model.decoder.layers[0](decode.clone())
    assert torch.equal(observed_decode, expected_decode)
    assert torch.equal(observed_decode[1], decode[1])

    runner.requests["on"] = _request(
        {"steer": {**on["steer"], "apply_at_all_positions": False}},
        output_token_ids=[7],
    )
    assert torch.equal(model.model.decoder.layers[0](decode.clone()), decode)

    runner.input_batch.req_ids = ["zero"]
    runner.requests = {"zero": _request(zero, output_token_ids=[9])}
    metadata.query_start_loc = torch.tensor([0, 1])
    zero_input = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    assert torch.equal(model.model.decoder.layers[0](zero_input.clone()), zero_input)

    runner.input_batch.req_ids = ["on"]
    runner.requests = {"on": _request(on, output_token_ids=[10])}
    on_input = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    assert torch.equal(model.model.decoder.layers[0](on_input.clone()), on_input + 2 * vector)
    runner.requests["on"].sampling_params.extra_args = None
    assert torch.equal(model.model.decoder.layers[0](on_input.clone()), on_input)


def test_activation_steer_effect_lifecycle(cache_dir, tmp_path):
    register_plugins()
    llm = HookLLM(
        model="gpt2",
        worker_name="steer_hook_act",
        download_dir=str(cache_dir),
        gpu_memory_utilization=0.1,
        max_model_len=64,
        dtype=torch.float16,
        enable_hook=True,
        enable_prefix_caching=False,
    )

    hidden_size = llm.llm.llm_engine.model_config.hf_config.hidden_size
    vector_path = tmp_path / "steering.pt"
    generator = torch.Generator().manual_seed(0)
    vector = torch.randn(hidden_size, generator=generator)
    vector /= vector.norm()
    torch.save({"dir": vector, "avg_proj": torch.tensor(0.0)}, vector_path)

    off = SamplingParams(temperature=0.0, max_tokens=3, logprobs=5, seed=0)
    on = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        logprobs=5,
        seed=0,
        extra_args={
            "steer": {
                "method": "add_vector",
                "coefficient": 100.0,
                "optimal_layer": 0,
                "vector_path": str(vector_path),
                "apply_at_all_positions": True,
            }
        },
    )
    prompt = "The capital of France is"

    off1 = llm.generate(prompt, sampling_params=off, use_hook=False)[0]
    on_output = llm.generate(prompt, sampling_params=on, use_hook=True)[0]
    off2 = llm.generate(prompt, sampling_params=off, use_hook=False)[0]

    off1_token_ids = tuple(off1.outputs[0].token_ids)
    on_token_ids = tuple(on_output.outputs[0].token_ids)
    off2_token_ids = tuple(off2.outputs[0].token_ids)
    off1_logprobs = _top_logprobs(off1)
    on_logprobs = _top_logprobs(on_output)
    off2_logprobs = _top_logprobs(off2)

    assert off2_token_ids == off1_token_ids
    assert off2_logprobs == off1_logprobs
    assert on_token_ids != off1_token_ids or on_logprobs != off1_logprobs
