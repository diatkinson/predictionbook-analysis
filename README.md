All the commands assume your current directory is the project root.

## Files
Data is stored in `data/`.

Jsonnet files for AllenNLP are in `experiments/`.

Explorations are in `notebooks/`.

Outputs are in `results/`.

## 1. How to scrape pages

The current `data/pages` directory contains pages that were last (fully) scraped in July 2019.

It was last updated in mid-September.

Updating is done by running `pba scrape --page-dir data/pages`which will download:

1) any newly created events, and
2) any existing and public, but unresolved events.

and stick them in `data/pages`. (Which is to say, if an event was judged when we scraped it, but was later marked unresolved, we won't update that page.)

If you want to scrape every page anew (which can take days), run `pba scrape --page-dir PAGEDIR --from-scratch`.

## 2. How to convert the raw pages into a json file

Run `pba gen-json data/pages data/predictions.json` (or whichever page directory and json output file you'd like.)

## 3. How to create train / validate / test splits

To split `data/predictions.json` into 80% train, 10% validation, and 10% test sets, run `pba split-dataset 80 10 10 data/predictions.json`, which will place the output files in the same directory as the json file you gave it. This only keeps predictions for which the outcome is already known, and it drops a random number of responses from each prediction (to simulate the standard situation of encountering a prediction somewhere in the middle of its lifecycle.) This will also take a subset of each train/valid/test file and copy it to toy files.
