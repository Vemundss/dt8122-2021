import argparse
import sys


def get_parameters():
    """Handle user inputs: dataset and method choice"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="yacht.txt", help="Which dataset to use", type=str
    )
    parser.add_argument("--method", default=0, help="Which method to use", type=int)
    args = parser.parse_args()

    dataset_name = args.dataset.split("/")[-1]  # dataset name - without path specs
    dataset_path = "../datasets/" + dataset_name

    try:
        f = open(dataset_path, "rb")
        f.close()
    except OSError:
        print(f"Could not open/read file with path: {dataset_path!r}. Exiting..")
        sys.exit()

    # Check if chosen method is contained in implemented methods wrt.
    # assignemnt numbering
    if args.method not in [0, 1, 2]:
        raise NotImplementedError

    return dataset_path, args.method


if __name__ == "__main__":
    """Simple check for get_parameters()"""
    dsp, method_num = get_parameters()
    print(f"Echo arguments: {dsp=} {method_num=}")
