# Hydro-STGAT

Physics-informed spatio-temporal graph attention network (**HYDRO-STGAT**) for daily streamflow in
river basins, in two stages:

1. **Natural regime**: a GRU + graph-attention model with a physically derived lagged-routing
   tensor (`M_lag`), trained on natural streamflow.
2. **Regulated regime**: the pretrained natural model is extended with a regulation module
   (water-use extraction and reservoir operation) by **transfer learning**, and validated
   against monthly gauged flows.

> Companion code of: `[Mena et al., "Hydro-STGAT: A physics-informed spatio-temporal graph attention network for natural and managed streamflow forecasting in regulated river basins", Environmental Modelling & Software, 2026]`.

```
 climate history ──► GRU ──► GAT ──┬─► local runoff r ─┐
                                   │                   ├─► routing with M_lag ──► Q_nat
 static attributes ────────────────┘   lag weights w ──┘          (notebook 01)
                                                                     │ pretrained weights
 water demand ──► RegulationModule (alpha, gamma) ──► Q_reg ◄────────┘   (notebook 02)
```

## Repository layout

```
hydro-stgat/
├── notebooks/
│   ├── 01_natural_regime_training.ipynb       # train + random search, natural regime
│   └── 02_regulated_transfer_learning.ipynb   # transfer learning, regulated regime
dataset: https://doi.org/10.5281/zenodo.23882217
```

## Reproducing the manuscript

1. Run **`01_natural_regime_training.ipynb`** from top to bottom. It writes
   `checkpoints/best_pi_stgat_lagged.pt` and
   `checkpoints/best_pi_stgat_lagged_config.json`.
   To skip training, copy the published weights and set `run_search = False` (and
   `eval_only = True`).
2. Run **`02_regulated_transfer_learning.ipynb`** from top to bottom. It reads the two files
   above, trains the regulation module (phase 1) and fine-tunes the whole model (phase 2).
   To skip training, copy the published regulated weights and set `eval_only = True`.

### Main settings

| Setting | Value | Where |
|---------|-------|-------|
| Transit velocity | 51.84 km/day (0.6 m/s) | notebook 01 `CFG["v_km_per_day"]` |
| Maximum lag | 30 days (31 lags) | both notebooks `CFG["max_lag"]` |
| Encoder window | 15 days (`window_gru`); history per sample = 45 days | both notebooks, **must be equal** |
| Split | chronological 80 / 10 / 10 % | both notebooks |
| Natural training | Adam, MSE on `log1p` flow, batch 32, ≤ 20 epochs, early stopping | notebook 01 |
| Random search | 10 trials: `lr`, `hidden_dim`, `num_heads`, `dropout`, `num_layers`; selected by mean validation NSE | notebook 01 |
| Regulated phase 1 | encoder frozen, lr 1e-3, 25 epochs | notebook 02 |
| Regulated phase 2 | lr 1e-5 (encoder) / 5e-4 (regulation), ≤ 75 epochs, patience 12 | notebook 02 |
| Loss weights | gauges 1.0, RREA simulation 0.05, physical penalty 0.1 | notebook 02 |

### Reproducibility notes

* Seeds (`seed = 42`) are set at the start of each notebook and again before the random
  search. Bit-for-bit identical results across different hardware or PyTorch versions are not
  guaranteed.
* The architecture of the transferred encoder is stored in
  `best_pi_stgat_lagged.json` and read by notebook 02, so both notebooks always
  use the same model. The notebook raises an error if `window_gru` or `max_lag` differ.
* In notebook 02 the whole model, including the frozen encoder, is put in `train()` mode in
  phase 1, so the dropout layers of the encoder remain active.
* The metric files use the columns `NSE, KGE, R2, RMSE_hm3_day, MAE_hm3_day, PBIAS_percent`.


## License

MIT, see `LICENSE`.

## Contact

`[Alberto Mena]` (`[amenvil@upv.es]`)
