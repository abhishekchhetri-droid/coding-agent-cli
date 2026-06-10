from agent import pipeline


def test_split_partitions_ok_and_ask():
    stages = [
        {"stage": "user input", "component": "ChatInput", "status": "ok"},
        {"stage": "intent classification", "status": "ask", "question": "Router or single Prompt?"},
        {"stage": "schema source", "status": "ask", "question": "Which Qdrant collection?"},
        {"stage": "output", "component": "ChatOutput", "status": "ok"},
    ]
    resolved, open_q = pipeline.split(pipeline.normalize_stages(stages))
    assert [s["stage"] for s in resolved] == ["user input", "output"]
    assert [q["stage"] for q in open_q] == ["intent classification", "schema source"]
    assert open_q[0]["question"] == "Router or single Prompt?"


def test_build_result_not_ready_when_questions_open():
    res = pipeline.build_result([], [{"stage": "x", "question": "y?"}])
    assert res["ready"] is False
    assert res["questions"][0]["question"] == "y?"
    assert "design_flow" in res["note"]  # told NOT to design yet


def test_build_result_ready_passes_resolved_stages():
    resolved = [{"stage": "user input", "component": "ChatInput", "status": "ok"}]
    res = pipeline.build_result(resolved, [])
    assert res["ready"] is True
    assert res["resolved_stages"] == resolved
    assert "design_flow" in res["note"]  # told to proceed


def test_ok_stage_without_component_demoted_to_ask():
    # An 'ok' status with no chosen component is contradictory — surface it as a question.
    stages = pipeline.normalize_stages([{"stage": "mystery", "status": "ok"}])
    assert stages[0]["status"] == "ask"
    assert "mystery" in stages[0]["question"]


def test_normalize_unknown_status_becomes_ask():
    stages = pipeline.normalize_stages([{"stage": "s", "component": "C", "status": "weird"}])
    assert stages[0]["status"] == "ask"


def test_normalize_drops_non_dict_and_blank_stage():
    stages = pipeline.normalize_stages(["nope", {"stage": "", "status": "ok"}, {"stage": "real", "component": "C", "status": "ok"}])
    assert [s["stage"] for s in stages] == ["real"]


def test_render_marks_ambiguous_with_question():
    out = pipeline.render_pipeline([
        {"stage": "user input", "component": "ChatInput", "source": "", "status": "ok"},
        {"stage": "schema", "status": "ask", "question": "Which collection?"},
    ])
    assert "`ChatInput`" in out
    assert "❓" in out and "Which collection?" in out


def test_render_shows_source_arrow():
    out = pipeline.render_pipeline([
        {"stage": "schema", "component": "QdrantVectorStore", "source": "schemas_collection", "status": "ok"},
    ])
    assert "QdrantVectorStore" in out and "schemas_collection" in out
