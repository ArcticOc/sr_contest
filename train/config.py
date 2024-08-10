import argparse

# 学習パラメーター
TRAIN_DATA_PATH = "dataset/train"
VAL_ORI_DATA_PATH = "dataset/validation/original"
VAL_025_DATA_PATH = "dataset/validation/0.25x"
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_WORKERS = 16
DEFAULT_NUM_EPOCH = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_ONLY_EVAL = False


def arg_parse():
    parser = argparse.ArgumentParser(description="ESPCN")
    parser.add_argument("--train-data-path", type=str, default=TRAIN_DATA_PATH, help="train data path")
    parser.add_argument(
        "--val-ori-data-path", type=str, default=VAL_ORI_DATA_PATH, help="validation original data path"
    )
    parser.add_argument("--val-025-data-path", type=str, default=VAL_025_DATA_PATH, help="validation 0.25x data path")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="number of workers")
    parser.add_argument("--num-epoch", type=int, default=DEFAULT_NUM_EPOCH, help="number of epochs")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="learning rate")
    parser.add_argument("--only-eval", action="store_true", default=DEFAULT_ONLY_EVAL, help="only eval")
    return parser.parse_args()


args = arg_parse()
