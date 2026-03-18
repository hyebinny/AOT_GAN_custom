#뗴오니가 만듬 260306
# masked, orig 돌리는 파일
# conda activate vitpose
# python vitpose_base.py

import cv2
import onepose
import os
import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

print("RUNNING FILE:", os.path.abspath(__file__))

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def collect_image_paths(root_dir: str):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if is_image_file(f):
                image_paths.append(os.path.join(root, f))
    return image_paths

def save_outputs_no_overwrite(
    keypoints,
    model,
    img_vis,
    image_path: str,
    input_base_dir: str,
    out_base_dir: str,
):
    rel_path = os.path.relpath(image_path, input_base_dir)
    rel_dir  = os.path.dirname(rel_path)

    pose_img_dir = os.path.join(out_base_dir, "pose_img", rel_dir)
    json_dir     = os.path.join(out_base_dir, "json", rel_dir)
    excel_dir    = os.path.join(out_base_dir, "excels", rel_dir)

    safe_makedirs(pose_img_dir)
    safe_makedirs(json_dir)
    safe_makedirs(excel_dir)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    save_image_path = os.path.join(pose_img_dir, f"vitpose_{base_name}.png")
    json_path       = os.path.join(json_dir,     f"vitpose_{base_name}.json")
    excel_path      = os.path.join(excel_dir,    f"vitpose_{base_name}.xlsx")

    if os.path.exists(save_image_path) and os.path.exists(json_path) and os.path.exists(excel_path):
        return (False, True)

    saved_anything = False

    if not os.path.exists(save_image_path):
        cv2.imwrite(save_image_path, img_vis)
        saved_anything = True

    if isinstance(keypoints, dict) and ('points' in keypoints) and ('confidence' in keypoints):
        points = keypoints['points']
        confidences = np.array(keypoints['confidence']).flatten()

        keypoints_list = []
        for i, (pt, conf) in enumerate(zip(points, confidences)):
            try:
                kp_info = model.keypoint_info[i]
                name = kp_info.get("name", f"keypoint_{i}") if isinstance(kp_info, dict) else f"keypoint_{i}"
            except Exception:
                name = f"keypoint_{i}"

            keypoints_list.append({
                "name": name,
                "x": float(pt[0]),
                "y": float(pt[1]),
                "confidence": float(conf)
            })

        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                json.dump({"keypoints": keypoints_list}, f, indent=2)
            saved_anything = True

        if not os.path.exists(excel_path):
            pd.DataFrame(keypoints_list).to_excel(excel_path, index=False)
            saved_anything = True

    return (saved_anything, False)

def run_vitpose_on_dir(model, input_base_dir: str, out_base_dir: str):
    safe_makedirs(out_base_dir)

    image_paths = collect_image_paths(input_base_dir)
    if len(image_paths) == 0:
        print(f"⚠️ No images found under: {input_base_dir}")
        return {
            "total": 0,
            "new_written": 0,
            "skipped_existing": 0,
            "failed": 0,
            "failed_list": []
        }

    failed_list = []
    new_written = 0
    skipped_existing = 0

    pbar = tqdm(image_paths, desc=f"ViTPose | {os.path.basename(input_base_dir)}", unit="img")
    for image_path in pbar:
        img = cv2.imread(image_path)
        if img is None:
            failed_list.append((image_path, "imread_failed"))
            pbar.set_postfix({"new": new_written, "skip": skipped_existing, "fail": len(failed_list)})
            continue

        try:
            keypoints = model(img)

            img_vis = img.copy()
            onepose.visualize_keypoints(
                img_vis,
                keypoints,
                model.keypoint_info,
                model.skeleton_info
            )

            saved_anything, skipped_all = save_outputs_no_overwrite(
                keypoints=keypoints,
                model=model,
                img_vis=img_vis,
                image_path=image_path,
                input_base_dir=input_base_dir,
                out_base_dir=out_base_dir
            )

            if skipped_all:
                skipped_existing += 1
            else:
                if saved_anything:
                    new_written += 1
                else:
                    skipped_existing += 1

        except Exception as e:
            failed_list.append((image_path, f"pose_failed: {repr(e)}"))

        pbar.set_postfix({"new": new_written, "skip": skipped_existing, "fail": len(failed_list)})

    return {
        "total": len(image_paths),
        "new_written": new_written,
        "skipped_existing": skipped_existing,
        "failed": len(failed_list),
        "failed_list": failed_list
    }

if __name__ == "__main__":
    input_root_dir = "/mnt/d/Hyebin/AOT-GAN-for-Inpainting/MAE-recon_results"
    out_root_dir   = "/mnt/d/Hyebin/AOT-GAN-for-Inpainting/MAE-pose_results"

    # 여기 적은 폴더만 실행
    target_dirs = [
        "ewha_masked/random_rec",
        "ewha_orig",
        "slp_masked/center_sq",
        "slp_masked/random_rec",
        "slp_masked/random_sq",
        "slp_orig",
    ]

    model = onepose.create_model()

    overall_start = time.time()
    summary_rows = []

    total_jobs = len(target_dirs)

    for job_idx, rel_dir in enumerate(target_dirs, start=1):
        input_base_dir = os.path.join(input_root_dir, rel_dir)
        out_base_dir   = os.path.join(out_root_dir, rel_dir, "vitpose")

        if not os.path.exists(input_base_dir):
            print(f"[{job_idx}/{total_jobs}] ⏭️ SKIP (missing): {input_base_dir}")
            continue

        print(f"\n==============================")
        print(f"[{job_idx}/{total_jobs}]")
        print(f"TARGET : {rel_dir}")
        print(f"IN  : {input_base_dir}")
        print(f"OUT : {out_base_dir}")
        print(f"NOTE: no overwrite mode (existing outputs are kept)")
        print(f"==============================\n")

        start = time.time()
        stats = run_vitpose_on_dir(model, input_base_dir, out_base_dir)
        elapsed = time.time() - start

        safe_makedirs(out_base_dir)
        fail_log_path = os.path.join(
            out_base_dir,
            f"failed_log_{rel_dir.replace('/', '_')}.csv"
        )
        pd.DataFrame(stats["failed_list"], columns=["image_path", "reason"]).to_csv(fail_log_path, index=False)

        print(
            f"✅ Done | total={stats['total']} new={stats['new_written']} "
            f"skip={stats['skipped_existing']} fail={stats['failed']} | {elapsed:.2f}s"
        )

        summary_rows.append({
            "target_dir": rel_dir,
            "total_images": stats["total"],
            "new_written": stats["new_written"],
            "skipped_existing": stats["skipped_existing"],
            "failed": stats["failed"],
            "elapsed_sec": elapsed,
            "input_dir": input_base_dir,
            "output_dir": out_base_dir,
            "failed_log": fail_log_path
        })

    overall_elapsed = time.time() - overall_start

    summary_path = os.path.join(out_root_dir, "vitpose_run_summary_selected_only.csv")
    safe_makedirs(os.path.dirname(summary_path))
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\n==============================")
    print(f"🏁 ALL DONE | total elapsed: {overall_elapsed:.2f}s")
    print(f"📄 Summary saved: {summary_path}")
    print("==============================\n")