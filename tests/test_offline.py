"""不依赖 MySQL、Redis、Milvus 等外部服务的回归测试。"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_VENV = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PROJECT_PYTHON = Path(
    os.getenv("EDERAG_PYTHON")
    or (_PROJECT_VENV if _PROJECT_VENV.is_file() else sys.executable)
)


def run_project_python(*args: str, code: str | None = None) -> subprocess.CompletedProcess:
    command = [str(PROJECT_PYTHON)]
    if code is not None:
        command.extend(["-c", code])
    command.extend(args)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )


def test_run_import_has_no_server_side_effect() -> None:
    result = run_project_python(code="import run; assert hasattr(run, 'uvicorn')")
    assert result.returncode == 0, result.stderr


def test_config_environment_override() -> None:
    code = """
import os
os.environ['EDU_MYSQL_PASSWORD'] = 'test-password'
os.environ['EDU_MILVUS_PORT'] = '19531'
from base.config import Config
config = Config()
assert config.MYSQL_PASSWORD == 'test-password'
assert config.MILVUS_PORT == 19531
"""
    result = run_project_python(code=code)
    assert result.returncode == 0, result.stderr


def test_document_ids_are_unique_and_stable() -> None:
    code = """
import importlib
import tempfile
from pathlib import Path
loader = importlib.import_module('2Milvus_RAG_Qa.core.document_loader')
with tempfile.TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)
    (root / 'a.md').write_text('# 标题\\n\\n第一份文档', encoding='utf-8')
    nested = root / 'nested'
    nested.mkdir()
    (nested / 'a.md').write_text('# 标题\\n\\n第二份文档', encoding='utf-8')
    parents1, children1 = loader.process_documents(str(root), parent_size=100, child_size=50, overlap=10)
    parents2, children2 = loader.process_documents(str(root), parent_size=100, child_size=50, overlap=10)
parent_ids = [item['id'] for item in parents1]
child_ids = [item['id'] for item in children1]
assert len(parent_ids) == len(set(parent_ids))
assert len(child_ids) == len(set(child_ids))
assert parent_ids == [item['id'] for item in parents2]
assert child_ids == [item['id'] for item in children2]
assert all(len(item_id) <= 64 for item_id in parent_ids + child_ids)
"""
    result = run_project_python(code=code)
    assert result.returncode == 0, result.stderr


def test_cli_help_does_not_start_services_or_training() -> None:
    commands = [
        ("MAIN.py", "--help"),
        ("-m", "1MySQL_qa.mysql_qa_main", "--help"),
    ]
    for command in commands:
        result = run_project_python(*command)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_architecture_contains_complete_executable_workflow() -> None:
    document = (PROJECT_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    required_commands = [
        "pip install -r requirements.txt",
        "python -m 1MySQL_qa.mysql_qa_main init-db",
        "python MAIN.py rebuild",
        "python MAIN.py query",
        "python run.py",
        "python -m 2Milvus_RAG_Qa.RAG评测.ragas_evaluate",
        "pytest -q",
        "pip check",
    ]
    assert document.count("```") % 2 == 0
    for command in required_commands:
        assert command in document
    assert "docker start milvus-etcd" not in document
    assert "sk-" not in document
    assert "X-API-Key" in document


def test_env_example_matches_config_variable_names() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "EDU_MILVUS_COLLECTION_NAME=" in example
    assert "EDU_APP_UPLOAD_API_KEY=" in example


def test_web_ui_and_upload_security_guards() -> None:
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "bubble.textContent=text" in source
    assert "X-API-Key" in source
    assert "hmac.compare_digest" in source
    assert "while chunk := await file.read" in source
    assert 'HTTPException(500, "知识库更新失败")' in source
    assert 'f"知识库更新失败: {exc}"' not in source
