import cv2
import onepose
import os
import json
import pandas as pd
import numpy as np
import time

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):

        return None

if __name__ == "__main__":

    # =========================================================
    # === (1) Set dir (ONLY CHANGE THESE TWO LINES) ============
    # =========================================================
    input_root_dir = "/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_recon_results"
    out_root_dir = "/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_pose_results"

    ckpt_iters = "500_000"
    dataset = "slp"
    mask_type = "center_sq"   # center_sq    random_sq     random_rec

    input_dir = f"{input_root_dir}/{ckpt_iters}/{dataset}/{mask_type}"
    base_out_dir = f"{out_root_dir}/{ckpt_iters}/{dataset}/{mask_type}/vitpose"
    # =========================================================
    
    image_save_dir = f"{base_out_dir}/pose_img"
    json_save_dir  = f"{base_out_dir}/json"
    excel_save_dir = f"{base_out_dir}/excels"

    for d in [image_save_dir, json_save_dir, excel_save_dir]:
        os.makedirs(d, exist_ok=True)

    # === (2) Load model once ===
    model = onepose.create_model()

    start_time = time.time()

    # === (3) Iterate over image files ===
    for filename in os.listdir(input_dir):
        if not is_image_file(filename):
            continue

        print(f"\n🖼️ Processing file: {filename}")
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"❌ Failed to read image: {image_path}")
            continue

        # === (4) Keypoint detection ===
        keypoints = model(img)

        # print("📌 Detected keypoints:")
        # print(keypoints)
        # print("📌 Type of keypoints:", type(keypoints))
        # if hasattr(keypoints, '__len__'):
        #     print("📌 Length of keypoints:", len(keypoints))

        # === (5) Visualize keypoints ===
        onepose.visualize_keypoints(img, keypoints, model.keypoint_info, model.skeleton_info)

        # === (6) Save visualized image ===
        save_image_name = f"vitpose_{filename}"
        save_image_path = os.path.join(image_save_dir, save_image_name)
        cv2.imwrite(save_image_path, img)
        print(f"✅ Saved image: {save_image_path}")

        # === (7) Prepare keypoint data ===
        if isinstance(keypoints, dict) and 'points' in keypoints and 'confidence' in keypoints:
            points = keypoints['points']
            confidences = keypoints['confidence'].flatten()

            keypoints_list = []
            for i, (pt, conf) in enumerate(zip(points, confidences)):
                name = model.keypoint_info[i]["name"] if "name" in model.keypoint_info[i] else f"keypoint_{i}"
                keypoints_list.append({
                    "name": name,
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "confidence": float(conf)
                })

            # === (8) Save JSON ===
            json_filename = f"vitpose_{os.path.splitext(filename)[0]}.json"
            json_path = os.path.join(json_save_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump({"keypoints": keypoints_list}, f, indent=2)
            print(f"📄 Saved JSON: {json_path}")

            # === (9) Save Excel ===
            excel_filename = f"vitpose_{os.path.splitext(filename)[0]}.xlsx"
            excel_path = os.path.join(excel_save_dir, excel_filename)
            df = pd.DataFrame(keypoints_list)
            df.to_excel(excel_path, index=False)
            print(f"📄 Saved Excel: {excel_path}")            

        else:
            print("⚠️ Unexpected keypoint format; skipping JSON/Excel export.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n⏱️ Total elapsed time: {:.2f} seconds".format(elapsed_time))