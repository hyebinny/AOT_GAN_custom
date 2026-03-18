import importlib
import os
from glob import glob

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from utils.option import args
from utils.painter import Sketcher


def postprocess(image):
    # tensor(RGB, [-1,1]) -> uint8 BGR (cv2용)
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # -> BGR
    return image


def demo(args):
    # load images
    img_list = []
    for ext in ["*.jpg", "*.png"]:
        img_list.extend(glob(os.path.join(args.dir_image, ext)))
    img_list.sort()

    # Model and version
    net = importlib.import_module("model." + args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    model = net.InpaintGenerator(args).to(device)
    model.load_state_dict(torch.load(args.pre_train, map_location=device))
    model.eval()

    for fn in img_list:
        filename = os.path.basename(fn).split(".")[0]

        # cv2는 BGR로 읽음 -> RGB로 변환해서 모델 입력과 맞춤
        orig_bgr = cv2.resize(cv2.imread(fn, cv2.IMREAD_COLOR), (256, 256))
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # 모델 입력은 RGB 기준으로 정규화
        img_tensor = (ToTensor()(orig_rgb) * 2.0 - 1.0).unsqueeze(0).to(device)

        h, w, c = orig_bgr.shape
        mask = np.zeros([h, w, 1], np.uint8)

        # UI(Sketcher)는 원래처럼 BGR로 보여주기
        image_copy = orig_bgr.copy()
        sketch = Sketcher(
            "input", [image_copy, mask], lambda: ((255, 255, 255), (255, 255, 255)), args.thick, args.painter
        )

        while True:
            ch = cv2.waitKey()
            if ch == 27:
                print("quit!")
                break

            # inpaint by deep model
            elif ch == ord(" "):
                print("[**] inpainting ... ")
                with torch.no_grad():
                    mask_tensor = (ToTensor()(mask)).unsqueeze(0).to(device)
                    masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
                    pred_tensor = model(masked_tensor, mask_tensor)
                    comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)

                    pred_np = postprocess(pred_tensor[0])      # BGR
                    masked_np = postprocess(masked_tensor[0])  # BGR
                    comp_np = postprocess(comp_tensor[0])      # BGR

                    cv2.imshow("pred_images", comp_np)
                    print("inpainting finish!")

            # reset mask
            elif ch == ord("r"):
                # reset도 RGB 기준으로 다시 텐서 만들기
                img_tensor = (ToTensor()(orig_rgb) * 2.0 - 1.0).unsqueeze(0)

                image_copy[:] = orig_bgr.copy()
                mask[:] = 0
                sketch.show()
                print("[**] reset!")

            # next case
            elif ch == ord("n"):
                print("[**] move to next image")
                cv2.destroyAllWindows()
                break

            elif ch == ord("k"):
                print("[**] apply existing processing to images, and keep editing!")
                img_tensor = comp_tensor                  # comp_tensor는 RGB 텐서
                image_copy[:] = comp_np.copy()            # comp_np는 BGR 이미지
                mask[:] = 0
                sketch.show()
                print("reset!")

            elif ch == ord("+"):
                sketch.large_thick()

            elif ch == ord("-"):
                sketch.small_thick()

            # save results
            if ch == ord("s"):
                cv2.imwrite(os.path.join(args.outputs, f"{filename}_masked.png"), masked_np)
                cv2.imwrite(os.path.join(args.outputs, f"{filename}_pred.png"), pred_np)
                cv2.imwrite(os.path.join(args.outputs, f"{filename}_comp.png"), comp_np)
                cv2.imwrite(os.path.join(args.outputs, f"{filename}_mask.png"), mask)

                print("[**] save successfully!")
        cv2.destroyAllWindows()

        if ch == 27:
            break


if __name__ == "__main__":
    demo(args)
