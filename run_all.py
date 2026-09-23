"""
Runs the full pipeline: install dependencies, train models, launch app.

Usage:
    python run_all.py
"""
import subprocess
import sys


def run(command):
    print(f"\n>>> Running: {' '.join(command)}\n")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\nERROR: command failed -> {' '.join(command)}")
        sys.exit(1)


if __name__ == "__main__":
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    run([sys.executable, "src/train.py"])
    run([sys.executable, "-m", "streamlit", "run", "app.py"])
