import subprocess
from datetime import datetime


def get_git_sha() -> str:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return sha
    except Exception:
        return "unknown"


def get_model_version_tag() -> str:
    sha = get_git_sha()
    date_str = datetime.now().strftime("%Y%m%d")
    return f"v0.1-{date_str}-{sha}"
