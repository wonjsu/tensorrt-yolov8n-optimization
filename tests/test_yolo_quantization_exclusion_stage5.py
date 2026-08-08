from examples.yolo_int8.run_quantization_exclusion_stage5 import (
    partition_cv2_branches,
    stage5_variants,
)


def test_partition_cv2_branches_exact_and_nonoverlapping():
    cv2 = [
        "/model.22/cv2.0/a",
        "/model.22/cv2.0/b",
        "/model.22/cv2.1/a",
        "/model.22/cv2.1/b",
        "/model.22/cv2.2/a",
        "/model.22/cv2.2/b",
    ]
    groups = partition_cv2_branches(cv2)
    assert list(groups) == ["cv2_0", "cv2_1", "cv2_2"]
    assert groups["cv2_0"] == ["/model.22/cv2.0/a", "/model.22/cv2.0/b"]
    assert groups["cv2_1"] == ["/model.22/cv2.1/a", "/model.22/cv2.1/b"]
    assert groups["cv2_2"] == ["/model.22/cv2.2/a", "/model.22/cv2.2/b"]
    assert set().union(*(set(value) for value in groups.values())) == set(cv2)


def test_stage5_variants_return_one_cv2_branch_to_quantization():
    groups = {
        "cv2": [
            "/model.22/cv2.0/a", "/model.22/cv2.0/b",
            "/model.22/cv2.1/a", "/model.22/cv2.1/b",
            "/model.22/cv2.2/a", "/model.22/cv2.2/b",
        ],
        "cv3": ["/model.22/cv3.0/a", "/model.22/cv3.1/a"],
        "dfl": ["/model.22/dfl/a"],
        "other": ["/model.22/other/a"],
    }
    branches = partition_cv2_branches(groups["cv2"])
    variants = stage5_variants(groups, branches)
    parent = set(groups["cv2"] + groups["cv3"] + groups["dfl"])

    assert set(variants) == {"quantize_cv2_0", "quantize_cv2_1", "quantize_cv2_2"}
    for label, excluded in variants.items():
        returned = set(branches[label.removeprefix("quantize_")])
        assert returned.isdisjoint(excluded)
        assert returned | set(excluded) == parent
        assert set(groups["other"]).isdisjoint(excluded)


def test_stage5_source_contains_no_latency_benchmark():
    import inspect
    import examples.yolo_int8.run_quantization_exclusion_stage5 as stage5

    source = inspect.getsource(stage5)
    assert "benchmark_precision" not in source
    assert "benchmark_trt_engines" not in source
