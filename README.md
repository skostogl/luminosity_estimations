# Luminosity estimations

Run an HL-LHC luminosity estimate by passing the YAML configuration file:

```bash
python3 example_hllhc.py config_2041.yaml
```

Results are written to `outputs/config_2041/`. The output folder contains a
copy of the configuration, `results.parquet`, and the generated plots.

To select a different folder:

```bash
python3 example_hllhc.py config_2041.yaml --output-dir /path/to/results
```
