from experience.models import ExperienceIngestRequest
from experience.writer import compute_context_hash


def test_compute_context_hash_stable():
    req1 = ExperienceIngestRequest(
        task_type="fix_bug",
        project="proj",
        repo="repo",
        stack={"python": "3.11", "framework": "fastapi"},
    )
    req2 = ExperienceIngestRequest(
        task_type="fix_bug",
        project="proj",
        repo="repo",
        stack={"framework": "fastapi", "python": "3.11"},
    )
    assert compute_context_hash(req1) == compute_context_hash(req2)


