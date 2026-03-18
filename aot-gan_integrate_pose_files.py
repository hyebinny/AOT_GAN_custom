#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

SRC_ROOT = Path("/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_pose_results_hrnet_openpose")
DST_ROOT = Path("/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_pose_results")

# 충돌 정책: "skip" 또는 "overwrite"
ON_CONFLICT = "skip"


def move_tree(src: Path, dst: Path):
    """
    Move all contents of src directory into dst directory.
    If dst exists, merge contents.
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        dst_item = dst / item.name

        if not dst_item.exists():
            shutil.move(str(item), str(dst_item))
            continue

        # conflict: both exist
        if item.is_dir() and dst_item.is_dir():
            move_tree(item, dst_item)
            # src/item might be empty now
            try:
                item.rmdir()
            except OSError:
                pass
        else:
            if ON_CONFLICT == "overwrite":
                if dst_item.is_dir():
                    shutil.rmtree(dst_item)
                else:
                    dst_item.unlink()
                shutil.move(str(item), str(dst_item))
            else:
                # skip: keep dst, remove src item
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC not found: {SRC_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # 찾을 것: SRC 아래의 "hrnet" 디렉토리와 "openpose_cvt" 디렉토리
    targets = []
    for p in SRC_ROOT.rglob("*"):
        if not p.is_dir():
            continue
        if p.name == "hrnet":
            targets.append((p, "hrnet"))
        elif p.name == "openpose_cvt":
            targets.append((p, "openpose"))

    targets.sort(key=lambda x: str(x[0]))
    print(f"[Found] {len(targets)} folders to move (hrnet + openpose_cvt)")

    for src_dir, new_name in targets:
        rel = src_dir.relative_to(SRC_ROOT)       # e.g. 000_000/ewha/random_rec/openpose_cvt
        dst_dir = DST_ROOT / rel.parent / new_name

        print(f"\n[MOVE]")
        print(f"  SRC: {src_dir}")
        print(f"  DST: {dst_dir}")

        move_tree(src_dir, dst_dir)

        # src_dir should be empty now; try removing
        try:
            src_dir.rmdir()
        except OSError:
            pass

    print("\n[Done]")

if __name__ == "__main__":
    main()
