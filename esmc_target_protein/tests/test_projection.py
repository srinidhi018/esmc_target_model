"""Optional projection: shapes, checkpoint requirement, order of operations."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from esmc_target.errors import ProjectionError
from esmc_target.projection import TargetProjection, load_projection, project_embeddings


def trained_checkpoint(tmp_path, input_dim=1152, output_dim=256, trained=True):
    projection = TargetProjection(input_dim, output_dim)
    # give the weights non-trivial values so order-of-operations is observable
    with torch.no_grad():
        projection.linear.weight.fill_(0.001)
        projection.linear.bias.fill_(0.002)
        projection.layer_norm.weight.fill_(1.0)
        projection.layer_norm.bias.fill_(0.0)
    path = tmp_path / "projection.pt"
    torch.save({"state_dict": projection.state_dict(),
                "metadata": {"trained": trained, "input_dim": input_dim,
                             "output_dim": output_dim, "source": "CancerCombo Scenario 3"}}, path)
    return path


def test_projection_shapes():
    projection = TargetProjection(1152, 256)
    assert projection.linear.in_features == 1152 and projection.linear.out_features == 256
    assert projection.layer_norm.normalized_shape == (256,)
    out = projection(torch.randn(1152))
    assert out.shape == (256,)
    assert isinstance(projection.layer_norm, nn.LayerNorm)


def test_projection_rejects_wrong_input_width():
    projection = TargetProjection(1152, 256)
    with pytest.raises(ProjectionError):
        projection(torch.randn(256))


def test_load_projection_requires_a_trained_checkpoint(tmp_path):
    path = trained_checkpoint(tmp_path, trained=True)
    projection = load_projection(path)
    assert projection.linear.in_features == 1152
    assert not any(p.requires_grad for p in projection.parameters())


def test_untrained_checkpoint_is_refused(tmp_path):
    path = trained_checkpoint(tmp_path, trained=False)
    with pytest.raises(ProjectionError) as excinfo:
        load_projection(path)
    assert "untrained" in str(excinfo.value).lower() or "not marked as trained" in str(excinfo.value)


def test_missing_checkpoint_is_fatal(tmp_path):
    with pytest.raises(ProjectionError) as excinfo:
        load_projection(tmp_path / "nope.pt")
    assert "random weights are never used" in str(excinfo.value).lower()


def test_dimension_mismatch_is_refused(tmp_path):
    path = trained_checkpoint(tmp_path, input_dim=1024)
    with pytest.raises(ProjectionError):
        load_projection(path, input_dim=1152)


def test_projection_applied_to_the_aggregate_not_per_protein():
    """Linear commutes with a mean, so project-after-mean == mean-of-projects
    (before LayerNorm). The pipeline must aggregate first."""
    torch.manual_seed(0)
    projection = TargetProjection(1152, 256)
    projection.eval()
    vectors = [torch.randn(1152) for _ in range(3)]
    aggregate_first = project_embeddings({"d": torch.stack(vectors).mean(dim=0)}, projection)["d"]
    per_protein = torch.stack([projection(v) for v in vectors]).mean(dim=0)
    # LayerNorm does not commute with averaging, so these are NOT equal; the
    # documented order (aggregate -> Linear -> LayerNorm) is the one enforced.
    assert not torch.allclose(aggregate_first, per_protein, atol=1e-4)
    lin = projection.linear
    mean_of_linear = lin(torch.stack(vectors).mean(dim=0))
    assert torch.allclose(aggregate_first, projection.layer_norm(mean_of_linear), atol=1e-6)
    assert torch.allclose(mean_of_linear, torch.stack([lin(v) for v in vectors]).mean(dim=0),
                          atol=1e-5)


def test_no_projection_training_entry_points_exist():
    import esmc_target.projection as projection
    source = open(projection.__file__, encoding="utf-8").read()
    assert "def train" not in source
    assert "save_checkpoint" not in source
    assert "optim" not in source
