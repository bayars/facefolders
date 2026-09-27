"""Assert the output of a run on make_photos.py fixtures. Usage: python check_output.py /out"""
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {r["file"]: r for r in csv.DictReader(open(out / "report.csv"))}
errors = []


def expect(cond, msg):
    if not cond:
        errors.append(msg)


people_dirs = [d for d in out.iterdir() if (d / ".person.json").exists()]
expect(len(people_dirs) == 6, f"expected 6 people, got {len(people_dirs)}")
for d in people_dirs:
    expect((d / "_face.jpg").exists(), f"{d.name}: missing _face.jpg")
    expect(len(json.loads((d / ".person.json").read_text())["centroid"]) == 512, f"{d.name}: bad marker")

for f in ["group.jpg", "group_small.jpeg", "sub/group.JPG", "big.jpg", "rotated.jpg"]:
    r = rows.get(f)
    expect(r and r["category"] == "people" and len(r["persons"].split(";")) == 6,
           f"{f}: expected all 6 people, got {r and (r['category'], r['persons'])}")
expect(rows.get("sky.jpg", {}).get("category") == "no_faces", "sky.jpg should be no_faces")
expect(rows.get("broken.jpg", {}).get("category") == "failed", "broken.jpg should be failed")
expect((out / "no_faces" / "sky.jpg").exists(), "sky.jpg not copied to no_faces/")
expect(any((d / "sub__group.JPG").exists() for d in people_dirs), "subfolder photo not flattened")

if errors:
    print("FAIL\n  " + "\n  ".join(errors))
    sys.exit(1)
print(f"OK: 6 people, {len(rows)} photos checked")
