# Reproducing the analysis of the CESAR1 campaign

This folder contains the files necessary for reproducing the results presented in "Deep learning for industrial processes: Forecasting amine emissions from a carbon capture plant".

## Dependencies

The exact environment we used to create the results is reflected in the `environment.yml` file.

Note that there have been some breaking changes in the darts library since we started our working. For this reason, please use our fork `darts @ git+https://github.com/kjappelbaum/darts.git@f15abd973ea72be76921ffcfc83908a2afaed912`. We will update the code in due course to be interoperable with the most recent version of darts.

## 1. Preprocessing

The original data was provided in the form of Excel files, which are deposited on [Zenodo](https://dx.doi.org/10.5281/zenodo.5153417).
To parse these files, one can use the `load_process_data` function in `parse_cesar1.py`.
This function is called for all the files, which are then concatenated in a dataframe in the notebook.
The output of this process is provided in the `20210624_df_cleaned.pkl` dataframe.

To reproduce the steps

- The notebook `eda_preprocessing.ipynb` contains the EDA and produces files (`detrended.pkl`) that are used after an additional processing step
- The notebook `gbdt_model.ipynb` applies a z-score filter (`z_score_filter` function from the preprocessing package), exponential window smoothing (`exponential_window_smoothing` function) and resampling to regular frequency of 2 min (`resample_regular` function).

## 2. Training the TCN models

We trained all notebooks from Jupyter notebooks in which we used the `run_model` function provided in `pyprocessta.model.tcn`.

- To produce the `train/test` plots (Figure 2 in the main text) we used the `validate_model.ipynb` notebook.
- To produce the models trained on all data (used for the heatmaps) we used the `train_model_on_all_data.ipynb` notebook. In order to use the models in the script for the heatmap creation, the models need to be converted, which we performed in the `export_models.ipynb` notebook.

## 3. Running the Causal Impact Analysis

For the Causal Impact Analysis models the timesteps of the changes as well as the dataframe with the process and emissions data are required.
To run the analysis, we used the `tcn_causalimpact.py` script. Depending on the input the following parameters need to be adjusted:

- `MEAS_COLUMNS` needs to reflect the features that are available for the model (even though they might not be used if they are Granger causally related with the intervention variable)
- `HORIZON` is the number of timesteps for which the forecast is made
- The `TARGETS` variable depends on how the model is supposed to be trained. We trained two models (combining AMP/PZ and ammonia/carbon dioxide). The name of this list must consist of the following allowed strings "2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2",
  "Carbon dioxide CO2", "Ammonia NH3"

The script assumes that `20210508_df_for_causalimpact.pkl` and `step_times.pkl` are in the same directory.

If all the parameters are set correctly, the script can be called as

```bash
python tcn_causalimpact.py
```

## 4. Creating the heatmaps

We ran the calculation of the heatmaps on a cluster that uses the `SLURM` scheduler. For this, we used the `loop_over_maps.py` script that creates a `SLURM` submission script for every heatmap. In this `SLURM` script the `run_scenarios` script is called. This script depends on the following inputs/customizations:

- `MEAS_COLUMNS` must equal to the `MEAS_COLUMNS` used when training the models
- `UPDATE_MAPPING` must contain the paths to the models and scalers that are to be used for the heatmaps
- `SCALER` is the scaler that was used to scale the covariates for the models mentioned in the line above

if these parameters are set correctly, the script can be called as

```bash
python run_scenarios.py {feature_a} {feature_b} {objective} {baseline}
```
