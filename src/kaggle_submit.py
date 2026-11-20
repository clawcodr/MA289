"""Kaggle client for the MA289 competition, using a bearer API token."""
import os, sys, time

KEY = open(os.path.expanduser("~/.kaggle/key.txt")).read().strip()
os.environ["KAGGLE_API_TOKEN"] = KEY
os.environ.setdefault("KAGGLE_USERNAME", "bearer")
os.environ.setdefault("KAGGLE_KEY", KEY)
COMP = "ma-289-ay-27-1-neural-network-word-embedding-competition"

import kaggle  # noqa: E402
api = kaggle.api

def submissions():
    return api.competition_submissions(COMP)

def show():
    rows = sorted(submissions(), key=lambda s: str(s.date))
    for s in rows:
        print(str(s.date)[:19], getattr(s, "status", None), "public=",
              getattr(s, "public_score", None), "|", getattr(s, "description", ""))
    print("TOTAL:", len(rows))
    return rows

def submit(path, message):
    api.competition_submit(path, message, COMP)
    print("uploaded:", os.path.basename(path), "|", message, flush=True)
    for _ in range(120):
        time.sleep(4)
        rows = sorted(submissions(), key=lambda s: str(s.date))
        s = rows[-1] if rows else None
        if s is None:
            continue
        st = str(getattr(s, "status", "")).upper()
        if "COMPLETE" in st and getattr(s, "public_score", None) is not None:
            print(f"SCORED {s.public_score}  | {s.description}", flush=True)
            return s
        if "ERROR" in st:
            print("ERROR:", getattr(s, "error_description", ""), flush=True)
            return s
    print("timeout waiting for score")

if __name__ == "__main__":
    if sys.argv[1] == "list":
        show()
    elif sys.argv[1] == "submit":
        submit(sys.argv[2], sys.argv[3])
