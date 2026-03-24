# spec-writer/app/scan.py
"""Scan the mounted codebase and show file type inventory."""
import os

SKIP_DIRS = {"bin", "obj", "node_modules", ".git", ".vs", "packages", "TestResults"}
CODEBASE = "/app/codebase"

exts = {}
total = 0

for root, dirs, files in os.walk(CODEBASE):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        exts[ext] = exts.get(ext, 0) + 1
        total += 1

print(f"\nCodebase: {CODEBASE}")
print(f"Total files: {total}\n")
print(f"  {'Extension':12s}   Count")
print(f"  {'-'*12:12s}   -----")
for ext, count in sorted(exts.items(), key=lambda x: -x[1]):
    print(f"  {ext or '(none)':12s}   {count}")