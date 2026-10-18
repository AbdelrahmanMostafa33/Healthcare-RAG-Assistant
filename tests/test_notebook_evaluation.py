"""Keep all four notebooks readable and the evaluation steps safe to run by default."""

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = [
    NOTEBOOK_DIR / "01_data_and_kb.ipynb",
    NOTEBOOK_DIR / "02_build_index.ipynb",
    NOTEBOOK_DIR / "03_rag_pipeline.ipynb",
    NOTEBOOK_DIR / "04_evaluation.ipynb",
]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_all_notebooks_start_with_a_title_and_share_the_same_kernel():
    for path in NOTEBOOKS:
        notebook = _read(path)
        first_cell = notebook["cells"][0]
        assert first_cell["cell_type"] == "markdown"
        assert "# 0" in "".join(first_cell["source"])
        assert notebook["metadata"]["kernelspec"]["display_name"] == ".venv (3.12.10)"


def test_notebooks_are_shipped_without_stale_execution_outputs():
    for path in NOTEBOOKS:
        for cell in _read(path)["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None


def test_evaluation_code_cells_compile():
    notebook = _read(NOTEBOOKS[-1])
    for index, source in enumerate(_code_cells(notebook)):
        compile(source, f"notebooks/04_evaluation.ipynb cell {index}", "exec")


def test_data_notebook_uses_the_dataset_question_focus_column():
    code = "\n".join(_code_cells(_read(NOTEBOOKS[0])))
    assert '"question_focus"' in code
    assert '"question_type"' not in code


def test_pipeline_walkthrough_handles_exhausted_transient_retries():
    notebook = _read(NOTEBOOKS[2])
    code = "\n".join(_code_cells(notebook))
    assert "ProviderTransientFailure" in code
    assert "LLM-only comparison" not in "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )


def test_calibration_is_local_and_paid_evaluation_is_opt_in():
    code = "\n".join(_code_cells(_read(NOTEBOOKS[-1])))
    assert "RUN_DEV_EVALUATION = False" in code
    assert "RUN_TEST_EVALUATION = False" in code
    assert "score_dev_questions(retriever, questions)" in code
    assert "calibration_from_scores(calibration_scores)" in code
    assert "run_pipeline_batch(pipeline, frame)" in code
    assert 'evaluate_split(df_test, "test")' in code
    assert "RUN_LLM_BASELINE" not in code
    assert "RunCheckpoint" not in code and "JudgeCheckpoint" not in code
