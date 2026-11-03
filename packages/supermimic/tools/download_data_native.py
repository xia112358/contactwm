"""Download the upstream motion/robot assets without Docker; resume existing files."""
from supermimic.paths import runtime_path
import concurrent.futures
import json
from pathlib import Path
import urllib.request

ROOT = runtime_path("data")
REPO = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
API = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main"


def listing(folder):
    with urllib.request.urlopen(f"{API}/{folder}?limit=1000", timeout=120) as response:
        return json.load(response)


def download(item):
    remote, destination = item
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{REPO}/resolve/main/{remote}", timeout=180) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            temporary.replace(destination)
            print(destination.relative_to(ROOT), flush=True)
            return
        except Exception:
            if attempt == 2:
                raise


if __name__ == "__main__":
    items = [(x["path"], ROOT / "lafan1_g1" / Path(x["path"]).name)
             for x in listing("g1") if x["type"] == "file" and x["path"].endswith(".csv")]
    items += [(x["path"], ROOT / "g1_description/meshes" / Path(x["path"]).name)
              for x in listing("robot_description/g1/meshes") if x["type"] == "file"]
    items.append(("robot_description/g1/g1_29dof_rev_1_0.urdf",
                  ROOT / "g1_description/g1_29dof_rev_1_0.urdf"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, items))
    clips = list((ROOT / "lafan1_g1").glob("*.csv"))
    assert len(clips) == 40, f"Expected 40 motions, found {len(clips)}"
    print(f"Downloaded {len(clips)} motion clips and robot assets.")
