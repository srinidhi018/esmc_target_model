"""Unit tests for original source row index preservation (Tests T, U)."""

import pandas as pd
from esmc_target.data import filter_rows, prepare_rows


def test_source_row_index_preserved_with_max_rows():
    df = pd.DataFrame({
        "nsc_id": ["740", "741", "742", "743", "744"],
        "sequence": ["ACDEF", "ACDEF", "ACDEF", "ACDEF", "ACDEF"],
    })
    
    # Apply --max-rows 2
    subset_df, applied = filter_rows(df, max_rows=2)
    assert len(subset_df) == 2
    assert list(subset_df.index) == [0, 1]

    records = prepare_rows(subset_df, special_residue_policy="keep_if_supported")
    assert len(records) == 2
    assert records[0].source_row_index == 0
    assert records[1].source_row_index == 1


def test_source_row_index_preserved_with_nsc_ids():
    df = pd.DataFrame({
        "nsc_id": ["NSC-10", "NSC-20", "NSC-30", "NSC-40", "NSC-50"],
        "sequence": ["ACDEF", "ACDEF", "ACDEF", "ACDEF", "ACDEF"],
    })
    
    # Filter for NSC-20 and NSC-40 (indices 1 and 3)
    subset_df, applied = filter_rows(df, nsc_ids=["NSC-20", "NSC-40"])
    assert len(subset_df) == 2
    assert list(subset_df.index) == [1, 3]

def test_non_contiguous_subset_pipeline_indexing(tmp_path):
    import torch
    from conftest import make_resolved
    from esmc_target.pipeline import PreprocessingPipeline
    from esmc_target.config import load_config

    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "nsc_id,drug_name,target_id,gene_symbol,uniprot_id,sequence,sequence_length\n"
        "NSC-1,Drug1,T1,G1,P1,ACDEFGHIKL,10\n"
        "NSC-2,Drug2,T2,G2,P2,MNPQRSTVWY,10\n"
        "NSC-3,Drug3,T3,G3,P3,ACDEFGHIKL,10\n"
        "NSC-4,Drug4,T4,G4,P4,ACDEFGHIKL,10\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "out_subset"
    resolved = make_resolved()
    config = load_config(None)

    # Filter non-contiguous subset: NSC-2 and NSC-4 (original file row indices 1 and 3)
    pipeline = PreprocessingPipeline(
        config=config,
        output_dir=out_dir,
        input_path=csv_file,
        nsc_ids=["NSC-2", "NSC-4"],
        resolved_model=resolved,
    )
    res = pipeline.run()

    # Inspect target_embeddings.pt artifact
    target_data = torch.load(res.target_embeddings_path, map_location="cpu", weights_only=False)
    row_indices = target_data["row_embedding_row_indices"].tolist()
    row_mask = target_data["row_valid_mask"].tolist()
    metadata = target_data["row_metadata"]

    assert row_mask == [True, True]
    assert row_indices == [1, 3]
    assert len(metadata) == 2
    assert metadata[0]["source_row_index"] == 1
    assert metadata[1]["source_row_index"] == 3

