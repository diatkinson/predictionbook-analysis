import argparse
import json
import os
import random
import sys

import pba.parse as parse
import pba.scrape as scrape
from pba.prediction import drop_some_responses

random.seed(80)


def split_dataset(args: argparse.Namespace) -> None:
    """
    Given a json database, remove any preditions that are as yet unknown. For each of the
    predictions that are left, drop a random number of responses. Finally, split the resulting
    predictions into train, test, and validation files, placed in the same directory as the json
    database. Also take a subset and copy them to toy files.
    """
    splits = (args.train_percent, args.valid_percent, args.test_percent)
    splits_msg = "The three splits must each be integers in [0, 100], and collectively sum to 100."
    assert sum(splits) == 100 and all(0 <= split <= 100 for split in splits), splits_msg

    print("Loading ")
    predictions = parse.load_json_to_predictions(args.json_file)
    task_predictions = [drop_some_responses(pred) for pred in predictions if pred.known()]
    assert all(p.known() for p in task_predictions)
    train_n, valid_n, _ = [round(split / 100 * len(task_predictions)) for split in splits]
    random.shuffle(task_predictions)
    train = task_predictions[:train_n]
    valid = task_predictions[train_n:train_n+valid_n]
    test = task_predictions[train_n+valid_n:]

    # Write outputs
    out_dir = os.path.dirname(args.json_file)
    for name, data in zip(["train", "valid", "test"], [train, valid, test]):
        with open(os.path.join(out_dir, f"{name}-predictions.json"), "w") as f:
            json.dump([pred.to_dict() for pred in data], f)
            print(f"{name}: {len(data)} predictions")
    # toy
    for name, data in zip(["train", "valid", "test"], [train[:10], valid[:5], test[:5]]):
        with open(os.path.join(out_dir, f"{name}-predictions-toy.json"), "w") as f:
            json.dump([pred.to_dict() for pred in data], f)
            print(f"{name} toy: {len(data)} predictions")


def gen_json(args: argparse.Namespace) -> None:
    """
    Run through all the pages in the given directory, parse them into predictions, and then dump
    the results into a json file.
    """
    with open(args.json_file, "w") as f:
        json.dump([p.to_dict() for p in parse.parse_pages(args.page_dir)], f)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    scrape_parser = subparsers.add_parser("scrape")
    scrape_parser.add_argument("--page-dir", required=True)
    scrape_parser.add_argument("--from-scratch", action='store_true')
    scrape_parser.set_defaults(func=scrape.scrape)

    split_dataset_parser = subparsers.add_parser("split-dataset")
    split_dataset_parser.add_argument("train_percent", type=int)
    split_dataset_parser.add_argument("valid_percent", type=int)
    split_dataset_parser.add_argument("test_percent", type=int)
    split_dataset_parser.add_argument("json_file", type=str)
    split_dataset_parser.set_defaults(func=split_dataset)

    gen_json_parser = subparsers.add_parser("gen-json")
    gen_json_parser.add_argument("page_dir", type=str)
    gen_json_parser.add_argument("json_file", type=str)
    gen_json_parser.set_defaults(func=gen_json)

    args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
