from __future__ import annotations

import hashlib

import numpy as np
import torch

from assim_lib.direct_dynamics_geometry_preflight import (
    _cell_area_proxy,
    _schedule_indices,
    _tensor_sha256,
)


def test_schedule_selects_distinct_fixed_last_slices():
    indices = _schedule_indices(365 * 24, updates=256)
    assert len(indices) == len(set(indices)) == 256
    assert indices[0] == 23
    assert indices[-1] == 365 * 24 - 1
    assert all(index % 24 == 23 for index in indices)


def test_tensor_hash_records_shape_dtype_and_exact_bytes():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert _tensor_sha256(value) == _tensor_sha256(value.clone())
    assert _tensor_sha256(value) != _tensor_sha256(value.reshape(2, 6))
    assert _tensor_sha256(value) != _tensor_sha256(value.to(torch.float64))
    expected = hashlib.sha256()
    expected.update(str((3, 4)).encode("ascii"))
    expected.update(str(torch.float32).encode("ascii"))
    expected.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    assert _tensor_sha256(value) == expected.hexdigest()


def test_cell_area_proxy_accepts_singleton_mask_and_pads(tmp_path):
    lat = np.linspace(70.0, 72.0, 12, dtype=np.float64).reshape(3, 4)
    lon = np.linspace(10.0, 14.0, 12, dtype=np.float64).reshape(3, 4)
    # Avoid a rank-one synthetic coordinate map with a zero Jacobian.
    lon = lon + np.arange(3, dtype=np.float64)[:, None] ** 2
    latitude_path = tmp_path / "lat.npy"
    longitude_path = tmp_path / "lon.npy"
    np.save(latitude_path, lat)
    np.save(longitude_path, lon)
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    area, evidence = _cell_area_proxy(
        latitude_path,
        longitude_path,
        mask,
        (4, 8),
        torch.device("cpu"),
    )
    assert area.shape == (1, 1, 4, 8)
    assert torch.all(area[..., :3, :4] > 0)
    assert torch.all(area[..., 3:, :] == 0)
    assert torch.all(area[..., :, 4:] == 0)
    assert evidence["kind"].startswith("spherical_center_coordinate_jacobian_proxy")
