
This directory contains only the code needed to train, evaluate, summarize, and plot the controlled ENSO forecasting experiments used in the manuscript.


- `dataset.py`, `train_v2.py`, `loss_v2.py`, and `utils.py`: data loading, controlled model training, evaluation, and skill metrics.
- `nino34_utils.py`: Niño 3.4 and warm-water-volume index utilities.
- `models/`: AtmosFormer (`AtmosFormer`), CNN, ConvLSTM, and Geoformer implementations.
- `configs/spb_core_claim_experiments.csv`: manuscript input-configuration matrix.
- `scripts/`: end-to-end experiment, event-hindcast, summary, and figure-generation commands.
- `requirements.txt`: Python dependencies.


Place the following files in a `data/` directory at the repository root before running the scripts:

- `cmip6_historical_1900_2014_global_mechanism.nc`
- `obs_1958_1978_global_mechanism.nc`
- `obs_1980_2025_global_mechanism.nc`

For Figure 1, provide an NMME ACC CSV through `--nmme_csv`; the expected columns are `lead_index`, `lead_month`, and one column per NMME member.


Train and summarize the manuscript configuration matrix:

```bash
bash scripts/run_spb_core_claim.sh
```

Useful controls include:

```bash
DRY_RUN=1 bash scripts/run_spb_core_claim.sh
RUN_MODELS=AtmosFormerSPB,ENSOCNN,ENSOConvLSTM,ENSOGeoformer bash scripts/run_spb_core_claim.sh
RUN_TRAINING=0 RUN_ANALYSIS=1 bash scripts/run_spb_core_claim.sh
```

Run the fixed-checkpoint event hindcasts and generate manuscript Figures 2–8:

```bash
bash scripts/run_spb_core_claim_figures.sh
```

Generate the reference comparison in Figure 1:

```bash
python scripts/plot_spb_reference_fig1.py --nmme_csv /path/to/nmme_acc.csv
```

Count model parameters:

```bash
python scripts/report_core_model_sizes.py
```

All output directories are created at runtime. No data, checkpoints, results, or figures are included here.
