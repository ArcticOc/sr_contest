from pathlib import Path

import cv2
from tqdm import tqdm


def calc_and_print_PSNR():
    input_image_dir = Path("dataset/validation/0.25x")
    output_image_dir = Path("output/img")
    original_image_dir = Path("dataset/validation/original")
    output_label = ["ESPCN", "NEAREST", "BILINEAR", "BICUBIC"]
    output_psnr = [0.0, 0.0, 0.0, 0.0]
    original_image_paths = list(original_image_dir.iterdir())
    for image_path in tqdm(original_image_paths):
        input_image_path = input_image_dir / image_path.relative_to(original_image_dir)
        output_iamge_path = output_image_dir / image_path.relative_to(original_image_dir)
        input_image = cv2.imread(str(input_image_path))
        original_image = cv2.imread(str(image_path))
        espcn_image = cv2.imread(str(output_iamge_path))
        output_psnr[0] += cv2.PSNR(original_image, espcn_image)
        h, w = original_image.shape[:2]
        output_psnr[1] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_NEAREST))
        output_psnr[2] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_LINEAR))
        output_psnr[3] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_CUBIC))
    with open("result.log", "w") as f:
        for label, psnr in zip(output_label, output_psnr, strict=False):
            print(f"{label}: {psnr / len(original_image_paths)}")
            f.write(f"{label}: {psnr / len(original_image_paths)}\n")
            f.write("-------------------------------\n")
