import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest
import torch
from comfy.cli_args import args
args.cpu = True

path = Path(__file__).resolve().parents[1] / 'nodes/minimax_h3_latent_upscaler_3d.py'
spec = importlib.util.spec_from_file_location('h3_upscaler', path)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
REAL_LOAD = u.mm.load_models_gpu
REAL_UNLOAD = u.mm.unload_model_and_clones

@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.manual_seed(17)
    network = u.LatentResizer3D(channels=32, in_blocks=1, out_blocks=1, temporal_every=0).eval()
    checkpoint = tmp_path / 'small.pth'
    torch.save(network.state_dict(), checkpoint)
    monkeypatch.setattr(u.folder_paths, 'get_full_path_or_raise', lambda *a: str(checkpoint))
    u.MODEL_CACHE.clear()
    load = Mock()
    unload = Mock()
    monkeypatch.setattr(u.mm, 'load_models_gpu', load)
    monkeypatch.setattr(u.mm, 'unload_model_and_clones', unload)
    return network, load, unload

def run(src, force=True):
    return u.MinimaxH3LatentUpscaler3D.execute(
        {'samples': src}, 'small.pth', {'mode': u.UpscaleMode.SCALE_BY, 'scale': 2},
        32, True, force, 'cpu', 'fp32')

def test_managed_loading_preserves_output_and_input(setup):
    network, load, unload = setup
    src = torch.randn(1, 24, 2, 2, 2)
    original = src.clone()
    mean, std = u._make_norm_tensors('cpu', torch.float32)
    with torch.inference_mode():
        expected = network((src - mean) / std, scale=2, target_size=(2, 4, 4), enable_chunking=True) * std + mean
    result = run(src).result[0]['samples']
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(src, original)
    load.assert_called_once()
    assert load.call_args.kwargs['memory_required'] > 0
    assert load.call_args.kwargs['force_full_load'] is True
    unload.assert_called_once()
    assert unload.call_args.args[0] is load.call_args.args[0][0]

def test_inference_exception_unloads_managed_model(setup, monkeypatch):
    _, load, unload = setup
    def fail(*a, **k):
        raise RuntimeError('simulated inference failure')
    monkeypatch.setattr(u.LatentResizer3D, 'forward', fail)
    with pytest.raises(RuntimeError, match='simulated inference failure'):
        run(torch.zeros(1, 24, 2, 2, 2))
    load.assert_called_once()
    unload.assert_called_once()

def test_retained_model_still_uses_manager_on_repeated_runs(setup):
    _, load, unload = setup
    src = torch.zeros(1, 24, 1, 2, 2)
    run(src, False)
    run(src, False)
    assert load.call_count == 2
    assert load.call_args_list[0].args[0][0] is load.call_args_list[1].args[0][0]
    unload.assert_not_called()


def test_loading_failure_runs_managed_cleanup(setup):
    _, load, unload = setup
    load.side_effect = RuntimeError('simulated load failure')
    with pytest.raises(RuntimeError, match='simulated load failure'):
        run(torch.zeros(1, 24, 1, 2, 2))
    unload.assert_called_once()


def test_no_resize_does_not_load_model(setup):
    _, load, unload = setup
    latent = {'samples': torch.zeros(1, 24, 2, 2)}
    result = u.MinimaxH3LatentUpscaler3D.execute(
        latent, 'small.pth', {'mode': u.UpscaleMode.SCALE_BY, 'scale': 1},
        32, True, True, 'cpu', 'fp32')
    assert result.result[0] is latent
    load.assert_not_called()
    unload.assert_not_called()


def test_real_comfy_manager_loads_and_unloads_cpu_model(setup, monkeypatch):
    monkeypatch.setattr(u.mm, 'load_models_gpu', REAL_LOAD)
    monkeypatch.setattr(u.mm, 'unload_model_and_clones', REAL_UNLOAD)
    result = run(torch.zeros(1, 24, 1, 2, 2)).result[0]['samples']
    assert result.shape == (1, 24, 1, 4, 4)
    assert torch.isfinite(result).all()
    patcher = next(iter(u.MODEL_CACHE.values()))
    assert not any(loaded.model is patcher for loaded in u.mm.current_loaded_models)
