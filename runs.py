"""Every run, including failed ones, is saved as JSON so it can be viewed and downloaded later."""
import json
import os
import re
import uuid

RUN_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def new_run_id():
    return uuid.uuid4().hex[:12]


def save_run(result, runs_dir):
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{result['run_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    return path


def load_run(run_id, runs_dir):
    """Return the stored run or None. The id is validated, so it can never name a path."""
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        return None
    path = os.path.join(runs_dir, f"{run_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
