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

## Peak luminosity scans

`scan_peak_lumi.py` evaluates the luminosity integrator of `lumi_class.LumiModel`
at a fixed beam state (no time evolution, no levelling) to compare optics,
crossing angles and crab cavity settings:

```bash
python3 scan_peak_lumi.py --config config_2036.yaml
```

Intensity, emittance, bunch length and number of bunches come from the
configuration unless given with `--intensity`, `--emittance-um`,
`--bunch-length-s` and `--bunches`. The crossing angle and crab cavities are in
the horizontal plane. By default the crossing angle follows a constant
long-range separation (`--lr-sep-sigma`, 10 sigma); `--xing-urad` fixes it
instead. The crab angle (`--crab-urad`, -190 urad) is capped at the crossing
angle unless `--no-crab-clip` is given. `python3 scan_peak_lumi.py --help`
lists the scan ranges and the remaining options.

Results are written to `outputs/peak_lumi_scan_<config name>/`:

- `peak_lumi_vs_sigmax*.png`, `scan.csv`: luminosity vs sigma*_x, one curve per
  sigma*_y (HL-LHC, HL-LHC without crab, LHC reference)
- `peak_lumi_vs_betax*.png`, `scan_beta.csv`: luminosity vs beta*_x, one curve per
  beta*_x/beta*_y, with the effective Piwinski angle
- `peak_lumi_vs_betax_betay<N>cm.png`, `scan_fixed_betay.csv`: luminosity and
  angles vs beta*_x at fixed beta*_y (both beams crabbed, B1 only, no crab)
- `peak_lumi_map_betax_xing_betay<N>cm*.png`, `scan_map_betax_xing.csv`:
  luminosity vs beta*_x and half crossing angle at fixed beta*_y, with
  long-range separation and Piwinski angle contours
- `peak_lumi_map*.png`, `scan_map.csv`: luminosity vs beta*_x/beta*_y and half
  crossing angle at constant long-range separation
- `scan_parameters.json`: the beam parameters and scan ranges used
