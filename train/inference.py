import datetime
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxoptimizer as optimizer
import onnxruntime as ort

from .config import args


class Qinference:
    def __init__(
        self,
        model_path=f"output/{args.model_type}/{args.model_type}.onnx",
        input_image_dir=args.val_025_data_path,
        output_image_dir="output/img",
    ):
        self.model_path = model_path
        self.input_image_dir = input_image_dir
        self.output_image_dir = output_image_dir

    def optimize_model(self):
        model = onnx.load(self.model_path)

        passes = [  # Optimizer passes for ResNet
            "fuse_bn_into_conv",
            "fuse_add_bias_into_conv",
            "fuse_pad_into_conv",
            "eliminate_nop_transpose",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
            "eliminate_nop_pad",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_concats",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_unused_initializer",
            "extract_constant_to_initializer",
            "fuse_consecutive_log_softmax",
            "eliminate_nop_monotone_argmax",
            "eliminate_deadend",
        ]

        optimized_model = optimizer.optimize(model, passes)

        optimized_model_path = self.model_path.replace(".onnx", "_optimized.onnx")
        onnx.save(optimized_model, optimized_model_path)
        print(f"Model has been optimized and saved to {optimized_model_path}")

        self.model_path = optimized_model_path

    def inference_onnxruntime(self):
        input_image_dir = Path(self.input_image_dir)
        output_image_dir = Path(self.output_image_dir)
        output_image_dir.mkdir(exist_ok=True, parents=True)

        print("Optimizing model...")
        self.optimize_model()

        sess = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_images = []
        output_images = []
        output_paths = []

        print("Loading images...")
        for image_path in input_image_dir.iterdir():
            output_image_path = output_image_dir / image_path.relative_to(input_image_dir)
            input_image = cv2.imread(str(image_path))
            input_image = (
                np.array([cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))], dtype=np.float32) / 255
            )
            input_images.append(input_image)
            output_paths.append(output_image_path)

        print("Running inference...")
        start_time = datetime.datetime.now()
        for input_image in input_images:
            output_images.append(sess.run(["output"], {"input": input_image})[0])
        end_time = datetime.datetime.now()

        print("Saving images...")
        for output_path, output_image in zip(output_paths, output_images, strict=False):
            output_image = cv2.cvtColor(
                (output_image.transpose((0, 2, 3, 1))[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(str(output_path), output_image)

        inference_time = (end_time - start_time).total_seconds() / len(input_images)
        print(f"Inference time: {inference_time:.4f} [s/image]")

        with open("result.log", "a") as f:
            f.write(f"Inference time: {inference_time:.4f} [s/image]\n")
