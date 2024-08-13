import datetime  # noqa: I001
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxoptimizer as optimizer
import onnxruntime as ort

from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from .config import args


class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_dir):
        self.image_paths = list(Path(calibration_image_dir).iterdir())
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        input_image = cv2.imread(str(image_path))
        input_image = (
            np.array([cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))], dtype=np.float32) / 255
        )

        return {"input": input_image}

    def rewind(self):
        self.index = 0


class Qinference:
    def __init__(
        self,
        model_path=args.model_path,
        quant_model_path=args.quant_model_path,
        input_image_dir=args.val_025_data_path,
        calibration_dir=args.val_ori_data_path,
        output_image_dir="output/img",
    ):
        self.model_path = model_path
        self.quant_model_path = quant_model_path
        self.input_image_dir = input_image_dir
        self.output_image_dir = output_image_dir
        self.calibration_dir = calibration_dir

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

    def static_quantize(self):
        self.optimize_model()
        calibration_data_reader = MyCalibrationDataReader(self.calibration_dir)
        quantize_static(
            self.model_path,
            self.quant_model_path,
            calibration_data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            reduce_range=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
        print(f"Model has been quantized and saved to {self.quant_model_path}")

    def inference_onnxruntime(self):
        infer_path = self.quant_model_path if args.quant_eval else self.model_path

        input_image_dir = Path(self.input_image_dir)
        output_image_dir = Path(self.output_image_dir)
        output_image_dir.mkdir(exist_ok=True, parents=True)
        if args.quant_eval:
            if Path(self.quant_model_path).exists():
                print(f"Quantized model already exists at {self.quant_model_path}. Skipping quantization.")
            else:
                print("Quantizing model...")
                self.static_quantize()
        else:
            print("Inference without quantization.")

        session_options = ort.SessionOptions()
        # session_options.log_severity_level = 1
        sess = ort.InferenceSession(
            infer_path, session_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
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
