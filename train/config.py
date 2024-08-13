import argparse

# 学習パラメーター
TRAIN_DATA_PATH = "dataset/train"
MODEL_PATH = "output/model.onnx"
VAL_ORI_DATA_PATH = "dataset/validation/original"
VAL_025_DATA_PATH = "dataset/validation/0.25x"
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_WORKERS = 16
DEFAULT_NUM_EPOCH = 100
DEFAULT_NUM_IMAGE_PER_EPOCH = 2000
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_ONLY_EVAL = False
DEFAULT_WRITER_NAME = "default"
DEFAULT_QUANT_MODEL_PATH = "output/model_quant.onnx"
DEFAULT_QUANT_EVAL = False


def arg_parse():
    parser = argparse.ArgumentParser(description="ESPCN")
    parser.add_argument("--train-data-path", type=str, default=TRAIN_DATA_PATH, help="train data path")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="model path")
    parser.add_argument(
        "--val-ori-data-path", type=str, default=VAL_ORI_DATA_PATH, help="validation original data path"
    )
    parser.add_argument("--val-025-data-path", type=str, default=VAL_025_DATA_PATH, help="validation 0.25x data path")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="number of workers")
    parser.add_argument("--num-epoch", type=int, default=DEFAULT_NUM_EPOCH, help="number of epochs")
    parser.add_argument(
        "--nipe",
        dest="num_image_per_epoch",
        type=int,
        default=DEFAULT_NUM_IMAGE_PER_EPOCH,
        help="number of images per epoch",
    )
    parser.add_argument("--lr", dest="learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="learning rate")
    parser.add_argument("--writer-name", type=str, default=DEFAULT_WRITER_NAME, help="writer name")
    parser.add_argument("--only-eval", action="store_true", default=DEFAULT_ONLY_EVAL, help="only eval")
    parser.add_argument("--quant-model-path", type=str, default=DEFAULT_QUANT_MODEL_PATH, help="quant model path")
    parser.add_argument("--quant-eval", action="store_true", default=DEFAULT_QUANT_EVAL, help="quant eval")
    return parser.parse_args()


args = arg_parse()
