import os
import sys

# Ensure the repository `src` (or repo root) is on sys.path so tests can import the package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")

# Prefer to add the `src` folder if it exists (common project layout), otherwise add repo root
if os.path.isdir(SRC_ROOT) and SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
elif REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
