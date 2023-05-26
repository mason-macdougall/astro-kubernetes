# Simplified astronomy-specific usage of Google Cloud's Kubernetes Engine

This is a simple toy-model to demonstrate how to run a parallelized set of many computationally-intensive Python jobs via Google Cloud's Kubernetes Engine.

## Folders and Files:
* Folder `bin/`
  * File `script.py`
    * Python script for a toy-model that creates synthetic Kepler-like lightcurve photometry and injects a transit signal
    * Takes commandline arguments:
      * `--sim_num`: a generic integer index that specifies the simulation number and the random kernel (integer > 0)
      * `--output_dir`: the preferred output directory
    * Saves the input transit parameters, lightcurve data, and a figure showing the injected transit
  * File `utils.py`
    * Includes basic imports and functions needed by `script.py`

* File `environment.yaml`
  * Specifies the Python environment needed to run `script.py` (e.g. from a `conda` environment)

* File `Snakemake`
  * Specifies how to `script.py` (i.e. inputs, outputs, number of iterations)
  * Specifies memory resources needed to run `script.py`
     * See https://snakemake.readthedocs.io/en/stable/ for more information

## Running on Kubernetes (via Snakemake)
* Full instructions available at: https://tinyurl.com/astrokube (Google Doc)
