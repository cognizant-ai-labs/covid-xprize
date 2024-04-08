"""
Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

This script creates plots for each stage of the LSTM pipeline. The plots are saved in the directory figures/. 
"""
import argparse
import time
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from covid_xprize.oxford_data.oxford_data import load_ips_file
from covid_xprize.oxford_data.npi_static import MAX_NPIS_VECTOR
from covid_xprize.examples.predictors.lstm.xprize_predictor import XPrizePredictor
from covid_xprize.validation.scenario_generator import ID_COLS, NPI_COLUMNS
from covid_xprize import oxford_data


def _most_affected_countries(df: pd.DataFrame, nb_geos: int, min_historical_days: int) -> list[str]:
    """
    Returns the list of most affected countries, in terms of confirmed deaths.
    :param df: the data frame containing the historical data
    :param nb_geos: the number of geos to return
    :param min_historical_days: the minimum days of historical data the countries must have
    :return: a list of country names of size nb_countries if there were enough, and otherwise a list of all the
    country names that have at least min_look_back_days data points.
    """
    # Don't include the region-level data, just the country-level summaries.
    gdf = df[df.RegionName.isna()]
    gdf = gdf.groupby('CountryName', group_keys=False)['ConfirmedDeaths'].agg(['max', 'count']).sort_values(
        by='max', ascending=False)
    filtered_gdf = gdf[gdf["count"] > min_historical_days]
    geos = list(filtered_gdf.head(nb_geos).index)
    return geos


def _write_ips_file_for_country(df: pd.DataFrame, country: str, start_date: datetime, end_date: datetime, file: str):
    gdf = df[
        (df.CountryName == country) &
        (df.RegionName.isna()) & # Don't include the region-level data, just the country-level summaries.
        (df.Date >= start_date) & (df.Date <= end_date)
    ]
    ips_df = gdf[ID_COLS + NPI_COLUMNS].copy()
    ips_df['RegionName'] = ips_df['RegionName'].fillna("")
    for npi_col in NPI_COLUMNS:
        ips_df.update(ips_df.groupby(['CountryName', 'RegionName'], group_keys=False)[npi_col].ffill().fillna(0))
    ips_df.to_csv(file, index=False)


def test_predictions_path(country, start_date_str) -> Path:
    directory = Path(__file__).parent / 'data' / 'predictions'; directory.mkdir(exist_ok=True)
    name_parts = []
    name_parts.append(f'country{country.replace(" ", "_")}')
    name_parts.append(start_date_str)
    return directory / ('_'.join(name_parts) + '.csv')


###############################################################################
# 06
###############################################################################
def test_train_predict_pipeline():
    """We want to train a model on a very small piece of data and confirm that
    the model is able to regurgitate the training data. If it is able to do so,
    then we know that in theory, the mode can learn time-series patterns seen
    in the data. This test is to establish basic pattern-recognition capabilities
    of the model. 
    """
    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_root_dir = current_dir / 'data'; data_root_dir.mkdir(exist_ok=True)
    data_dir = data_root_dir / '06_train_predict_pipeline'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Prepare some metadata.
    country = "Aruba" # Start with 
    ips_path = data_dir / f"ips_file.csv"
    figure_name = f"06_train_predict_pipeline"

    # Initialize an untrained model.
    df = oxford_data.load_original_oxford_data()
    path_to_oxford_data = data_dir / f"OxCGRT_artificial.csv"
    df.to_csv(path_to_oxford_data, index=False)
    model = XPrizePredictor(None, path_to_oxford_data)

    # Prepare artificial training data by modifying the model's dataframe in-place.
    model.df = model.df[model.df.GeoID == country].copy()
    df = model.df
    df[NPI_COLUMNS] = 0
    # The pattern is: 
    # 1. 14 days of NPIs==1.0, and case ratio equal to 1
    # 2. 28 days of NPIs==1.0, and case ratio equal to 0.9
    # 3. 14 days of NPIs==0.0, and case ratio equal to 1
    # 4. 28 days of NPIs==0.0, and case ratio equal to 1 / 0.9
    pattern_period = 14 + 28 + 14 + 28
    for row_i in range(len(df)):
        pattern_phase = row_i % pattern_period
        if pattern_phase < 14:
            df.loc[row_i, "PredictionRatio"] = 1.0
            df.loc[row_i, NPI_COLUMNS] = 1.0
        elif pattern_phase < 14 + 28:
            df.loc[row_i, "PredictionRatio"] = 0.9
            df.loc[row_i, NPI_COLUMNS] = 1.0
        elif pattern_phase < 14 + 28 + 14:
            df.loc[row_i, "PredictionRatio"] = 1.0
            df.loc[row_i, NPI_COLUMNS] = 0.0
        else:
            df.loc[row_i, "PredictionRatio"] = 1 / 0.9
            df.loc[row_i, NPI_COLUMNS] = 0.0

    # Get the start date from the data
    start_date = df.iloc[pattern_period + 14].Date
    n_prediction_days = 31
    end_date = start_date + timedelta(days=n_prediction_days-1)

    # Train the model on our artificial training data.
    model.train()

    # Create the prediction vectors for each scenario.
    prediction_ratios_by_npi_const: dict[float, np.ndarray] = {}
    initial_context_by_npi_const: dict[float, np.ndarray] = {}
    npis_vals = [0.0, 1.0]
    for npis_val in tqdm.tqdm(npis_vals, desc="Predictions (scenarios)"):
        # Get the initial context & action vectors.
        context, action = model._initial_context_action_vectors([country], start_date)
        context = context[country] # shape (Days, 1)
        action  = action [country] # shape (Days, NPIs)

        # Edit the action & context vector to match our scenario. 
        action [:] = npis_val
        context[:] = 1.0

        # Get the predictions with the passed NPIs
        npis_sequence = np.full((n_prediction_days, action.shape[1]), npis_val) # shape (PredictionDates, NPIs)
        pred_ratios = model._roll_out_predictions( # shape (PredictionDates,)
            model.predictor, 
            context,
            action,
            npis_sequence)

        # Add result to collection.
        prediction_ratios_by_npi_const[npis_val] = pred_ratios
        initial_context_by_npi_const[npis_val] = context.flatten()

    # Compile the results in to a figure.
    n_rows = 3
    n_cols = 2
    fig = plt.figure(figsize=(5 * n_cols, 2 * n_rows))
    do_legend = True

    ax = plt.subplot(n_rows, n_cols, 1)
    ax.axis("off")
    ax.text(0.1, 0.5, "\n".join([
        figure_name,
        "PredictionRatios are the immediate output of the v1 LSTM model."
    ]), in_layout=False)

    plot_i = n_cols + 1
    for npi_val in tqdm.tqdm(npis_vals, desc="Plots (Scenarios)"):
        # predictions = oxford_data.process_submission(predictions, start_date, end_date)
        prediction_ratios = prediction_ratios_by_npi_const[npi_val]
        initial_context = initial_context_by_npi_const[npi_val]

        # data = df[(df.Date >= start_date) & (df.Date <= end_date) & (df.GeoID == country)]
        # data = data.copy()
        # data["PredictedPredictionRatio"] = prediction_ratios

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        sns.set_palette(sns.husl_palette(8))
        plot = sns.lineplot(prediction_ratios, legend=do_legend, label="Predicted", color="black", ax=ax)
        # plot = sns.lineplot(data, x="Date", y="PredictedPredictionRatio",
        #                     legend=do_legend, label="Predicted PredictionRatio", color="red", ax=ax)
        # plot = sns.lineplot(data, x="Date", y="PredictionRatio",
        #                     legend=do_legend, label="Ground Truth PredictionRatio", color="black", ax=ax)
        # plot.get_figure().autofmt_xdate()
        ax.axhline(1.0, color="red", linestyle='--') # Draw a line at identity.
        ax.set_ylabel("")
        ax.set_ylim(0.8, 1.2)
        ax.set_title(f"NPIs={npi_val}")

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        sns.set_palette(sns.husl_palette(8))
        plot = sns.lineplot(initial_context, legend=do_legend, label="Initial Context", color="green", ax=ax)
        ax.set_ylabel("")
        ax.set_ylim(0.9, 1.2)
        ax.set_title(f"NPIs={npi_val}")
        if do_legend:
            # Put the lower left of the bounding box at axes coordinates (0, 1.20).
            sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1.20))
            do_legend = False


    plt.tight_layout()
    figure_path = figures_dir / f"{figure_name}.png"
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


###############################################################################
# 05
###############################################################################
    
def test_artificial_data(path_to_model_weights: str, start_date_str: str):
    _test_artificial_data_helper(path_to_model_weights, start_date_str, context_const=0.95)
    _test_artificial_data_helper(path_to_model_weights, start_date_str, context_const=1.05)


def _test_artificial_data_helper(path_to_model_weights: str, start_date_str: str, context_const: float):
    """We want to better understand the factors that the model takes into account.
    So, we will create artificial data in which individual parameters are changed,
    and generate predictions to evaluate the model sensitivitiy to these parameters. 
    :param path_to_model_weights: The model used for the evaluation. 
    :param start_date_str: The start date of the prediction period.
    """
    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_root_dir = current_dir / 'data'; data_root_dir.mkdir(exist_ok=True)
    data_dir = data_root_dir / '05_artificial_data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Prepare the original data.
    df = oxford_data.load_original_oxford_data()
    # df = df[df.Date < start_date] # Truncate the data if we're using it for training.
    path_to_oxford_data = data_dir / f"OxCGRT_orig.csv"
    df.to_csv(path_to_oxford_data, index=False)

    # Load the given model weights.
    model = XPrizePredictor(path_to_model_weights, path_to_oxford_data)

    # Prepare some metadata. 
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    n_prediction_days = 31
    end_date = start_date + timedelta(days=n_prediction_days-1)
    country = "France" # Start with 
    ips_path = data_dir / f"ips_file.csv"
    figure_name = f"05_artificial_data_{start_date_str}_context_{context_const}"

    # Prepare a base IPs df which we will modify to create the artificial data.
    _write_ips_file_for_country(
        df,
        country,
        start_date,
        end_date,
        ips_path,
    )
    npis_df = load_ips_file(ips_path)
    npis_sequence = np.array(npis_df[NPI_COLUMNS]) # shape (PredictionDates, NPIs)

    # Create the artificial test scenarios. 
    test_scenarios: list[tuple[str, np.ndarray]] = [] # pairs (label, npis)
    n_npis = len(MAX_NPIS_VECTOR)
    test_scenarios.append(("all_zero", np.zeros((n_prediction_days, n_npis)))) # shape (PredictionDates, NPIs)
    test_scenarios.append(("all_max",  np.full((n_prediction_days, n_npis), MAX_NPIS_VECTOR))) # shape (PredictionDates, NPIs)
    for npi_i, npi_label in enumerate(tqdm.tqdm(NPI_COLUMNS, desc="Make scenarios")):
        npis = npis_sequence.copy() # shape (PredictionDates, NPIs)
        npis[:, npi_i] = 0.0
        test_scenarios.append((f"{npi_label}_zero", npis))
        npis = npis_sequence.copy() # shape (PredictionDates, NPIs)
        npis[:, npi_i] = MAX_NPIS_VECTOR[npi_i]
        test_scenarios.append((f"{npi_label}_max", npis))
    labels = [scenario[0] for scenario in test_scenarios]
    assert(len(set(labels)) == len(labels)) # Make sure there's no duplicate labels.

    # Create the prediction vectors for each scenario.
    prediction_ratios_by_label: dict[str, np.ndarray] = {}
    for scenario_label, scenario in tqdm.tqdm(test_scenarios, desc="Predictions (scenarios)"):
        # Check the cache for already-made results.
        pred_ratios_cache_path = data_dir / f"{scenario_label}_{start_date_str}_{context_const}.npy"
        if pred_ratios_cache_path.exists():
            prediction_ratios_by_label[scenario_label] = np.load(pred_ratios_cache_path)
            continue

        # Get the initial context & action vectors.
        context, action = model._initial_context_action_vectors([country], start_date)
        context = context[country] # shape (Days, 1)
        action  = action [country] # shape (Days, NPIs)

        # Edit the action context vector to match our scenario by repeating the first day backwards. 
        action[:, :] = scenario[0:1, :] # broadcast shape (Days, NPIs) <- shape (1, NPIs)

        # Edit the context vector to fill in our data.
        context[:] = context_const

        # Get the predictions with the passed NPIs
        npis_sequence = scenario # shape (PredictionDates, NPIs)
        pred_ratios = model._roll_out_predictions( # shape (PredictionDates,)
            model.predictor, 
            context,
            action,
            npis_sequence)

        # Add result to cache.
        np.save(pred_ratios_cache_path, pred_ratios)

        # Add result to collection.
        prediction_ratios_by_label[scenario_label] = pred_ratios

    # Compute some additional columns on the GT data including prediction ratios. 
    df = oxford_data.prepare_cases_dataframe()

    # Compile the results in to a figure.
    n_scenarios = len(test_scenarios)
    n_rows = n_scenarios // 2 + 2
    n_cols = 2
    fig = plt.figure(figsize=(5 * n_cols, 2 * n_rows))
    do_legend = True

    ax = plt.subplot(n_rows, n_cols, 1)
    ax.axis("off")
    ax.text(0.1, 0.5, "\n".join([
        figure_name,
        "PredictionRatios are the immediate output of the v1 LSTM model."
    ]), in_layout=False)

    plot_i = n_cols + 1
    for label, scenario in tqdm.tqdm(test_scenarios, desc="Plots (Scenarios)"):
        # predictions = oxford_data.process_submission(predictions, start_date, end_date)
        prediction_ratios = prediction_ratios_by_label[label]

        # data = df[(df.Date >= start_date) & (df.Date <= end_date) & (df.GeoID == country)]
        # data = data.copy()
        # data["PredictedPredictionRatio"] = prediction_ratios

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        sns.set_palette(sns.husl_palette(8))
        plot = sns.lineplot(prediction_ratios, legend=do_legend, label="Predicted", color="black", ax=ax)
        # plot = sns.lineplot(data, x="Date", y="PredictedPredictionRatio",
        #                     legend=do_legend, label="Predicted PredictionRatio", color="red", ax=ax)
        # plot = sns.lineplot(data, x="Date", y="PredictionRatio",
        #                     legend=do_legend, label="Ground Truth PredictionRatio", color="black", ax=ax)
        # plot.get_figure().autofmt_xdate()
        ax.axhline(1.0, color="red", linestyle='--') # Draw a line at identity.
        ax.set_ylabel("")
        ax.set_ylim(0.9, 1.2)
        ax.set_title(label)
        if do_legend:
            # Put the lower left of the bounding box at axes coordinates (0, 1.20).
            sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1.20))
            do_legend = False

    plt.tight_layout()
    figure_path = figures_dir / f"{figure_name}.png"
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


###############################################################################
# 04
###############################################################################


def _test_plot_prediction_ratios_helper(path_to_model_weights: str, start_date_str: str, on_training_data: bool = False):
    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_root_dir = current_dir / 'data'; data_root_dir.mkdir(exist_ok=True)
    data_dir = data_root_dir / '06_prediction_ratios'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Prepare the original data.
    df = oxford_data.load_original_oxford_data()
    # df = df[df.Date < start_date] # Truncate the data if we're using it for training.
    path_to_oxford_data = data_dir / f"OxCGRT_orig.csv"
    df.to_csv(path_to_oxford_data, index=False)

    # Load the given model weights.
    model = XPrizePredictor(path_to_model_weights, path_to_oxford_data)

    # Prepare some metadata. 
    if on_training_data: # The training evaluations should be done on earlier data.
        pred_start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        start_date: datetime = pred_start_date - timedelta(days=60)
        start_date_str = start_date.strftime('%Y-%m-%d')
    else:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = start_date + timedelta(days=30)
    countries = _most_affected_countries(df, 30, 50)
    ips_path = data_dir / f"ips_file.csv"

    tr_or_st_label = "tr" if on_training_data else "st"
    figure_name = f"04_prediction_ratios_{start_date_str}_{tr_or_st_label}"

    # Create the prediction vectors for each country.
    prediction_ratios_by_country: dict[str, np.ndarray] = {}
    for country_i, country in enumerate(tqdm.tqdm(sorted(countries), desc="Predictions (Countries)")):
        # Check the cache for already-made results.
        pred_ratios_cache_path = data_dir / f"{country}_{start_date_str}.npy"
        if pred_ratios_cache_path.exists():
            prediction_ratios_by_country[country] = np.load(pred_ratios_cache_path)
            continue

        # Get the initial context & action vectors.
        context, action = model._initial_context_action_vectors([country], start_date)
        context = context[country] # shape (Days, 1)
        action  = action [country] # shape (Days, NPIs)

        # Write the true intervention plan during this time period.
        _write_ips_file_for_country(
            df,
            country,
            start_date,
            end_date,
            ips_path,
        )
        npis_df = load_ips_file(ips_path)

        # Prepare the DF and the NPIs for the country.
        # cdf = df[(df.CountryName == country) & (df.RegionName.isna())]
        # cdf = cdf[cdf.ConfirmedCases.notnull()]
        cnpis_df = npis_df[npis_df.GeoID == country] # TODO double check this isn't empty
        npis_sequence = np.array(cnpis_df[NPI_COLUMNS]) # shape (PredictionDates, NPIs)
        # Get the predictions with the passed NPIs
        pred_ratios = model._roll_out_predictions( # shape (PredictionDates,)
            model.predictor, 
            context,
            action,
            npis_sequence)

        # Add result to cache.
        np.save(pred_ratios_cache_path, pred_ratios)

        # Add result to collection.
        prediction_ratios_by_country[country] = pred_ratios

    # Compute some additional columns on the GT data including prediction ratios. 
    df = oxford_data.prepare_cases_dataframe()

    # Compile the results in to a figure.
    n_countries = len(countries)
    n_rows = int(np.floor(np.sqrt(n_countries))) + 2
    n_cols = int(np.ceil(n_countries / n_rows))
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))
    do_legend = True

    ax = plt.subplot(n_rows, n_cols, 1)
    ax.axis("off")
    ax.text(0.1, 0.5, "\n".join([
        figure_name,
        "PredictionRatios are the immediate output of the v1 LSTM model."
    ]), in_layout=False)

    plot_i = n_cols + 1
    for country in tqdm.tqdm(sorted(countries), desc="Plots (Countries)"):
        # predictions = oxford_data.process_submission(predictions, start_date, end_date)
        prediction_ratios = prediction_ratios_by_country[country]

        data = df[(df.Date >= start_date) & (df.Date <= end_date) & (df.GeoID == country)]
        data = data.copy()
        data["PredictedPredictionRatio"] = prediction_ratios

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        sns.set_palette(sns.husl_palette(8))
        # plot = sns.lineplot(prediction_ratios, legend=do_legend, label="Predicted", color="red", ax=ax)
        plot = sns.lineplot(data, x="Date", y="PredictedPredictionRatio",
                            legend=do_legend, label="Predicted PredictionRatio", color="red", ax=ax)
        plot = sns.lineplot(data, x="Date", y="PredictionRatio",
                            legend=do_legend, label="Ground Truth PredictionRatio", color="black", ax=ax)
        plot.get_figure().autofmt_xdate()
        ax.set_ylabel("")
        ax.set_ylim(0.9, 1.2)
        ax.set_title(country)
        if do_legend:
            # Put the lower left of the bounding box at axes coordinates (0, 1.20).
            sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1.20))
            do_legend = False

    plt.tight_layout()
    figure_path = figures_dir / f"{figure_name}.png"
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


def test_plot_prediction_ratios(path_to_model_weights: str, start_date_str: str):
    """Plot the immediate output of the neural network against the true data."""
    _test_plot_prediction_ratios_helper(path_to_model_weights, start_date_str, on_training_data=False)
    _test_plot_prediction_ratios_helper(path_to_model_weights, start_date_str, on_training_data=True)



###############################################################################
# 03
###############################################################################


def test_plot_input_output_pipeline(path_to_model_weights: str, start_date_str: str):
    """Plot the inputs to the neural network along with the raw prediction ratio outputs."""
    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_dir = current_dir / 'data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Prepare the original data.
    df = oxford_data.load_original_oxford_data()
    # df = df[df.Date < start_date] # Truncate the data if we're using it for training.
    path_to_oxford_data = data_dir / f"OxCGRT_orig.csv"
    df.to_csv(path_to_oxford_data, index=False)

    # Load the given model weights.
    model = XPrizePredictor(path_to_model_weights, path_to_oxford_data)

    # Prepare some related parameters.
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = start_date + timedelta(days=30)
    countries = _most_affected_countries(df, 30, 50)
    ips_path = data_dir / f"ips_file.csv"

    # Compile the results in to a figure.
    n_countries = len(countries)
    n_country_plots = int(np.floor(np.sqrt(n_countries)))
    n_cols = int(np.ceil(n_countries / n_country_plots))
    n_rows = 2 * n_country_plots + 2
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))
    plot_i = n_cols + 1
    do_legend = True

    # Create the context vectors for each country and plot them.
    for country_i, country in tqdm.tqdm(enumerate(sorted(countries)), desc="test_plot_input_output_pipeline (Countries)", total=len(countries)):
        # Get the initial context & action vectors.
        context, action = model._initial_context_action_vectors([country], start_date)
        context = context[country] # shape (Days, 1)
        action  = action [country] # shape (Days, NPIs)

        # Write the true intervention plan during this time period.
        _write_ips_file_for_country(
            df,
            country,
            start_date,
            end_date,
            ips_path,
        )
        npis_df = load_ips_file(ips_path)

        # Prepare the DF and the NPIs for the country.
        # cdf = df[(df.CountryName == country) & (df.RegionName.isna())]
        # cdf = cdf[cdf.ConfirmedCases.notnull()]
        cnpis_df = npis_df[npis_df.GeoID == country] # TODO double check this isn't empty
        npis_sequence = np.array(cnpis_df[NPI_COLUMNS]) # shape (PredictionDates, NPIs)
        # Get the predictions with the passed NPIs
        pred_ratios = model._roll_out_predictions(model.predictor, # shape (PredictionDates,)
                                                  context,
                                                  action,
                                                  npis_sequence)

        # Plot the predictions.
        row = 1 + 2 * (country_i // n_cols) # Skip the first row, then 2 rows per country.
        col = country_i % n_cols
        plot_i = 1 + n_cols * row + col
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        plt.subplots_adjust(hspace=0)
        sns.set_palette(sns.husl_palette(8))
        plot = sns.lineplot(
            x=cnpis_df.Date,
            y=pred_ratios, 
            color="red", legend=do_legend, label="Prediction Ratios", ax=ax)
        plot.get_figure().autofmt_xdate()
        ax.set_ylabel("")
        ax.set_title(country)
        ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.8) # Show the identity prediction ratio.
        # Also show the 2% increase and decrease ratios. (Remark: these two are not actually inverse operations.)
        r = 0.02
        ax.axhline(y=1+r, color='blue', linestyle='--',  alpha=0.4) 
        ax.axhline(y=1-r, color='blue', linestyle='--',  alpha=0.4)

        if do_legend:
            # Put the lower left of the bounding box at axes coordinates (0, 1.20).
            sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1.20))
            do_legend = False

        # Plot the NPIs.
        # Normalize the NPIs by dividing by the max values for correct color scaling.
        npis_max_v = np.array(MAX_NPIS_VECTOR) # shape (NPIs)
        npis_image = npis_sequence / npis_max_v # shape (PredictionDates, NPIs)
        row = 1 + 1 + 2 * (country_i // n_cols) # Skip the first row, then 2 rows per country.
        col = country_i % n_cols
        plot_i = 1 + n_cols * row + col
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        plt.subplots_adjust(hspace=0)
        ax.imshow(npis_image.T, cmap='inferno')

    # Plot some debug information
    plot_i = 1
    ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
    debug = "\n".join([
        f"test_plot_pred_ratios: Prediction Ratios output along with plus/minus {r:.2} marks (blue dashes).",
        f"Prediction Ratios are the raw output of the V1 LSTM-based neural network.",
        f"Below is the NPIs, divided by the max possible value in each NPI.",
    ])
    ax.axis("off")
    ax.text(0, 1, debug)

    figure_path = figures_dir / f"03_pred_ratios_{start_date}.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


###############################################################################
# 02
###############################################################################


def test_plot_initial_context(path_to_model_weights: str, start_date_str: str):

    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_dir = current_dir / 'data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Prepare the original data.
    df = oxford_data.load_original_oxford_data()
    # df = df[df.Date < start_date] # Truncate the data if we're using it for training.
    path_to_oxford_data = data_dir / f"OxCGRT_orig.csv"
    df.to_csv(path_to_oxford_data, index=False)

    # Load the given model weights.
    model = XPrizePredictor(path_to_model_weights, path_to_oxford_data)

    # Prepare some related parameters.
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    countries = _most_affected_countries(df, 30, 50)

    # Compile the results in to a figure.
    n_countries = len(countries)
    n_rows = int(np.floor(np.sqrt(n_countries))) + 2
    n_cols = int(np.ceil(n_countries / n_rows))
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))
    plot_i = n_cols + 1
    do_legend = True

    # Create the context vectors for each country and plot them.
    for country in tqdm.tqdm(sorted(countries), desc="test_plot_initial_context (Countries)"):
        # Get the initial context & action vectors.
        context, action = model._initial_context_action_vectors([country], start_date)
        context = context[country] # shape (Days, 1)
        action  = action [country] # shape (Days, NPIs)

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        sns.set_palette(sns.husl_palette(8))
        plot = sns.lineplot(context[:, 0], color="red", legend=do_legend, label="Context", ax=ax)
        # plot = sns.lineplot(gt         , x="Date", y="ActualDailyNewCases7DMA", legend=False, color="black", ax=ax)
        plot.get_figure().autofmt_xdate()
        ax.set_ylabel("")
        ax.set_title(country)

        if do_legend:
            # Put the lower left of the bounding box at axes coordinates (0, 1.20).
            sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1.20))
            do_legend = False

    figure_path = figures_dir / f"02_context_vectors_{start_date}.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


###############################################################################
# 01
###############################################################################


def test_plot_predictions(path_to_model_weights: str, start_date_str: str):
    print(f"test_plot_predictions:")
    print(f"    path_to_model_weights: {path_to_model_weights}")
    print(f"    start_date_str: {start_date_str}")

    _test_plot_predictions_helper(path_to_model_weights, start_date_str, on_training_data=False)
    _test_plot_predictions_helper(path_to_model_weights, start_date_str, on_training_data=True)


def _test_plot_predictions_helper(path_to_model_weights: str, start_date_str: str, on_training_data=False):
    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_dir = current_dir / 'data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Prepare the original data.
    df = oxford_data.load_original_oxford_data()
    # df = df[df.Date < start_date] # Truncate the data if we're using it for training.
    path_to_oxford_data = data_dir / f"OxCGRT_orig.csv"
    df.to_csv(path_to_oxford_data, index=False)

    # Load the given model weights.
    model = XPrizePredictor(path_to_model_weights, path_to_oxford_data)

    # Create the model predictions. 
    if on_training_data: # The training evaluations should be done on earlier data.
        pred_start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        start_date: datetime = pred_start_date - timedelta(days=60)
        start_date_str = start_date.strftime('%Y-%m-%d')
    else:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = start_date + timedelta(days=30)
    countries = _most_affected_countries(df, 30, 50)
    ips_path = data_dir / f"ips_file.csv"
    for country in tqdm.tqdm(countries, desc="Predictions (Countries)"):
        # Skip if already predicted.
        predictions_path = test_predictions_path(country, start_date_str)
        if predictions_path.exists():
            continue

        # Write the true intervention plan during this time period.
        _write_ips_file_for_country(
            df,
            country,
            start_date,
            end_date,
            ips_path,
        )

        # Evaluate the model predicitons.
        preds_df = model.predict(start_date, end_date, ips_path)
        preds_df.to_csv(predictions_path, index=False)

    ###########################################################################
    # Compile the results in to a figure.
    n_countries = len(countries)
    n_rows = int(np.floor(np.sqrt(n_countries))) + 2
    n_cols = int(np.ceil(n_countries / n_rows))
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))
    plot_i = n_cols + 1
    do_legend = True
    for country in tqdm.tqdm(sorted(countries), desc="Plots (Countries)"):
        predictions_path = test_predictions_path(country, start_date_str)
        predictions = pd.read_csv(predictions_path, parse_dates=['Date'])
        predictions = oxford_data.process_submission(predictions, start_date, end_date)

        gt_start = start_date - timedelta(days=14) # Show some additional context. 
        gt = df[(df.Date >= gt_start) & (df.Date <= end_date) & (df.CountryName == country) & (df.RegionName.isna())]
        gt = oxford_data.process_submission(gt, gt_start, end_date)

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        sns.set_palette(sns.husl_palette(8))
        plot = sns.lineplot(predictions, x="Date", y="PredictedDailyNewCases7DMA", legend=do_legend, label="Predicted", color="red", ax=ax)
        plot = sns.lineplot(gt         , x="Date", y="ActualDailyNewCases7DMA", legend=do_legend, label="Ground Truth", color="black", ax=ax)
        plot.get_figure().autofmt_xdate()
        ax.set_ylabel("")
        ax.set_title(country)
        if do_legend:
            # Put the lower left of the bounding box at axes coordinates (0, 1.20).
            sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1.20))
            do_legend = False

    tr_or_st_label = "tr" if on_training_data else "st"
    figure_path = figures_dir / f"01_final_predictions_{start_date_str}_{tr_or_st_label}.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


def main(path_to_model_weights: str, start_date_str: str):
    """
    :param path_to_model_weights: Path to the XPrizePredictor model weights.
    :param start_date_str: Date in the format %Y-%m-%d.  """
    print(f"Test pipeline beginning with the following arguments:")
    print(f"    path_to_model_weights: {path_to_model_weights}")
    print(f"    start_date_str: {start_date_str}")
    test_plot_predictions(path_to_model_weights, start_date_str)
    test_plot_initial_context(path_to_model_weights, start_date_str)
    test_plot_input_output_pipeline(path_to_model_weights, start_date_str)
    test_plot_prediction_ratios(path_to_model_weights, start_date_str)
    test_artificial_data(path_to_model_weights, start_date_str)
    test_train_predict_pipeline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_model_weights",
                        help="path to LSTM weights",
                        default="/Users/964353/PandemicResilience/Cognizant/models/scenario2_lstm_trained_model_weights.h5")
    
    # Scenario 2: Argentina, 2021-03-1 - 2021-04-31
    # Scenario 4: Kenya, 2020-10-01 - 2020-11-30
    parser.add_argument("--start_date_str",
                        help="start date of predictions",
                        default="2021-03-01")
    args = parser.parse_args()
    main(**vars(args))


