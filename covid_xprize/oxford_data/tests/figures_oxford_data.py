"""
Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

Supplement the tests with visual figures to verify the system is working correctly."""

import sys
import os
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from covid_xprize import oxford_data


###############################################################################
# 01
###############################################################################

def test_plot_case_columns():
    _test_plot_case_columns_helper("2020-11-01", "2020-11-30")
    _test_plot_case_columns_helper("2021-11-01", "2021-11-30")


def _test_plot_case_columns_helper(start_date, end_date):
    """Plot the various case columns."""

    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_dir = current_dir / 'data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Test cases show the input / output relationships of our method.
    test_cases: list[tuple[np.ndarray]] = []

    # Create some test cases using real data.
    df = oxford_data.prepare_cases_dataframe()
    n_test_countries = 15
    real_data_test_countries = oxford_data.most_affected_geos(df, n_test_countries, 30)
    for test_country in real_data_test_countries:
        cdf       = df[(df['GeoID'] == test_country) & (df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # Converting back from CaseRatio should equals the original column NewCases
        test_cases.append((
            test_country, cdf
        ))

    # Initialize the figure.
    n_cols = 8
    n_rows = 1 + len(test_cases)
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))
    ax = plt.subplot(n_rows, n_cols, 1)
    ax.axis("off")
    ax.text(0.1, 0.5, "\n".join([
        "test_plot_case_columns",
        "The real case columns and computed columns.",
    ]), in_layout=False)

    for row, (test_country, cdf) in enumerate(test_cases):
        label_titles = row == 0
        plot_i = ((row + 1) * n_cols) + 1

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.axis('off')
        ax.text(0.0, 0.5, test_country)
        for column_name, color in (
            ("ConfirmedCases", "green"),
            ("NewCases", "blue"),
            ("SmoothNewCases", "black"),
            ("SmoothNewCasesPer100K", "orange"),
            ("CaseRatio", "black"),
            ("ProportionInfected", "black"),
            ("PredictionRatio", "purple"),
        ):
            ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
            plot = sns.lineplot(data=cdf, x="Date", y=column_name, ax=ax, color=color)
            plot.get_figure().autofmt_xdate()
            if label_titles:
                ax.set_title(column_name)


    figure_path = figures_dir / f"01_case_columns_{start_date}.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


###############################################################################
# 02
###############################################################################


def running_mean_convolve(x, N):
    return np.convolve(x, np.ones(N) / float(N), 'valid')


def test_convert_prediction_ratios():
    """Test the conversion routines going "backwards"."""

    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_dir = current_dir / 'data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Test cases show the input / output relationships of our method.
    test_cases: list[tuple[np.ndarray]] = []

    # Create some test cases using real data.
    df = oxford_data.prepare_cases_dataframe()
    n_test_countries = 7
    real_data_test_countries = oxford_data.most_affected_geos(df, n_test_countries, 30)
    PRIOR_DATA_START       = "2021-10-01"
    PREDICTIONS_START_DATE = "2021-11-01"
    PREDICTIONS_END_DATE   = "2021-11-30"
    for test_country in real_data_test_countries:
        cdf       = df[(df['GeoID'] == test_country) & (df['Date'] >= PREDICTIONS_START_DATE) & (df['Date'] <= PREDICTIONS_END_DATE)]
        cdf_prior = df[(df['GeoID'] == test_country) & (df['Date'] >= PRIOR_DATA_START)       & (df['Date'] < PREDICTIONS_START_DATE)]

        # Converting back from PredictionRatio should equals the original column NewCases
        pop_size = cdf['Population'].max()
        curr_total_cases = cdf_prior['ConfirmedCases'].to_numpy()[-1]
        ratios = cdf['PredictionRatio'].to_numpy()
        prev_new_cases = cdf_prior['NewCases'].to_numpy()
        new_cases_reconstructed = oxford_data.convert_prediction_ratios_to_new_cases(
            ratios,
            7,
            prev_new_cases,
            curr_total_cases,
            pop_size,
        )
        original_new_cases = cdf['NewCases'].to_numpy()
        original_smooth_new_cases = cdf['SmoothNewCases'].to_numpy()
        test_cases.append((
            test_country, ratios, new_cases_reconstructed, original_new_cases, original_smooth_new_cases,
        ))

    # Create a test case using fake data.
    # Some common variables
    for curr_total_cases in (100_000, 1_000_000):
        for population in (10_000_000,):
            for prev_new_cases_c in (2_000, 20_000):
                label = f"curr_total_cases: {curr_total_cases:,}\npopulation:{population:,}\nprev_new_cases:{prev_new_cases_c:,}"
                prev_new_cases = np.full((20,), prev_new_cases_c)

                # Example 1: Exponential decay
                ratios = np.full((31,), 0.95)
                new_cases_reconstructed = oxford_data.convert_prediction_ratios_to_new_cases(
                    ratios,
                    7,
                    prev_new_cases,
                    curr_total_cases,
                    population
                )
                test_cases.append((
                    label, ratios, new_cases_reconstructed, None, None
                ))

                # Example 2: Exponential growth
                ratios = np.full((31,), 1.05)
                new_cases_reconstructed = oxford_data.convert_prediction_ratios_to_new_cases(
                    ratios,
                    7,
                    prev_new_cases,
                    curr_total_cases,
                    population
                )
                test_cases.append((
                    label, ratios, new_cases_reconstructed, None, None
                ))

    # Initialize the figure.
    n_cols = 7
    n_rows = 1 + len(test_cases)
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))

    for row, (label, gt_prediction_ratio, computed_new_cases, gt_new_cases, gt_smooth_cases) in enumerate(test_cases):
        label_titles = row == 0
        plot_i = ((row + 1) * n_cols) + 1

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.axis('off')
        ax.text(0.1, 0.5, label)
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.plot(gt_prediction_ratio, color="purple")
        if label_titles:
            plt.title("Actual PredictionRatio")
        ax = ax_output = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.plot(computed_new_cases, color="blue")
        if label_titles:
            plt.title("Computed NewCases")
        if gt_new_cases is not None:
            ax = plt.subplot(n_rows, n_cols, plot_i, sharey=ax_output); plot_i += 1
            ax.plot(gt_new_cases, color='blue')
            if label_titles:
                plt.title("Actual NewCases")
            ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
            ax.plot(gt_new_cases - computed_new_cases, color='red')
            if label_titles:
                plt.title("Residuals (Actual - Computed)")
        plot_i = ((row + 1) * n_cols) + n_cols - 1 # final-1 column of the row
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        dma = running_mean_convolve(computed_new_cases, 7)
        ax.plot(dma, color="black")
        if label_titles:
            plt.title("Computed SmoothNewCases")
        if gt_smooth_cases is not None:
            ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
            ax.plot(gt_smooth_cases[6:], color="black")
            if label_titles:
                plt.title("Actual SmoothNewCases")

    figure_path = figures_dir / f"02_convert_prediction_ratios.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


###############################################################################
# 03
###############################################################################


def running_mean_convolve(x, N):
    return np.convolve(x, np.ones(N) / float(N), 'valid')


def test_convert_smooth_new_cases_per_100k():
    """Test the conversion routines going "backwards"."""

    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_dir = current_dir / 'data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Test cases show the input / output relationships of our method.
    test_cases: list[tuple[np.ndarray]] = []

    # Create some test cases using real data.
    df = oxford_data.prepare_cases_dataframe()
    n_test_countries = 7
    real_data_test_countries = oxford_data.most_affected_geos(df, n_test_countries, 30)
    PRIOR_DATA_START       = "2021-10-01"
    PREDICTIONS_START_DATE = "2021-11-01"
    PREDICTIONS_END_DATE   = "2021-11-30"
    for test_country in real_data_test_countries:
        cdf       = df[(df['GeoID'] == test_country) & (df['Date'] >= PREDICTIONS_START_DATE) & (df['Date'] <= PREDICTIONS_END_DATE)]
        cdf_prior = df[(df['GeoID'] == test_country) & (df['Date'] >= PRIOR_DATA_START)       & (df['Date'] < PREDICTIONS_START_DATE)]

        # Converting back from PredictionRatio should equals the original column NewCases
        pop_size = cdf['Population'].max()
        smooth_cases_per_100k = cdf['SmoothNewCasesPer100K'].to_numpy()
        prev_new_cases = cdf_prior['NewCases'].to_numpy()
        new_cases_reconstructed = oxford_data.convert_smooth_cases_per_100K_to_new_cases(
            smooth_cases_per_100k,
            7,
            prev_new_cases,
            pop_size,
        )
        original_new_cases = cdf['NewCases'].to_numpy()
        original_smooth_new_cases = cdf['SmoothNewCases'].to_numpy()
        test_cases.append((
            test_country, smooth_cases_per_100k, new_cases_reconstructed, original_new_cases, original_smooth_new_cases,
        ))

    # Initialize the figure.
    n_cols = 7
    n_rows = 1 + len(test_cases)
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))

    for row, (label, gt_smooth_cases_per_100k, computed_new_cases, gt_new_cases, gt_smooth_cases) in enumerate(test_cases):
        label_titles = row == 0
        plot_i = ((row + 1) * n_cols) + 1

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.axis('off')
        ax.text(0.1, 0.5, label)
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.plot(gt_smooth_cases_per_100k, color="orange")
        if label_titles:
            plt.title("Actual SmoothNewCasesPer100K")
        ax = ax_output = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.plot(computed_new_cases, color="blue")
        if label_titles:
            plt.title("Computed NewCases")
        if gt_new_cases is not None:
            ax = plt.subplot(n_rows, n_cols, plot_i, sharey=ax_output); plot_i += 1
            ax.plot(gt_new_cases, color='blue')
            if label_titles:
                plt.title("Actual NewCases")
            ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
            ax.plot(gt_new_cases - computed_new_cases, color='red')
            if label_titles:
                plt.title("Residuals (Actual - Computed)")
        plot_i = ((row + 1) * n_cols) + n_cols - 1 # final-1 column of the row
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        dma = running_mean_convolve(computed_new_cases, 7)
        ax.plot(dma, color="black")
        if label_titles:
            plt.title("Computed SmoothNewCases")
        if gt_smooth_cases is not None:
            ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
            ax.plot(gt_smooth_cases[6:], color="black")
            if label_titles:
                plt.title("Actual SmoothNewCases")

    figure_path = figures_dir / f"03_convert_smooth_new_cases_per_100k.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


###############################################################################
# 04
###############################################################################

def running_mean_convolve(x, N):
    return np.convolve(x, np.ones(N) / float(N), 'valid')


def test_backward_pass():
    """Starting with conversion ratios, go step by step backwards."""

    # Prepare some paths for output and working data. 
    current_dir = Path(__file__).parent
    plots_dir = current_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    data_dir = current_dir / 'data'; data_dir.mkdir(exist_ok=True)
    figures_dir = current_dir / 'figures'; figures_dir.mkdir(exist_ok=True)

    # Test cases show the input / output relationships of our method.
    test_cases: list[tuple[np.ndarray]] = []

    # Create some test cases using real data.
    df = oxford_data.prepare_cases_dataframe()
    n_test_countries = 7
    real_data_test_countries = oxford_data.most_affected_geos(df, n_test_countries, 30)
    PRIOR_DATA_START       = "2021-10-01"
    PREDICTIONS_START_DATE = "2021-11-01"
    PREDICTIONS_END_DATE   = "2021-11-30"
    for test_country in real_data_test_countries:
        cdf       = df[(df['GeoID'] == test_country) & (df['Date'] >= PREDICTIONS_START_DATE) & (df['Date'] <= PREDICTIONS_END_DATE)]
        cdf_prior = df[(df['GeoID'] == test_country) & (df['Date'] >= PRIOR_DATA_START)       & (df['Date'] < PREDICTIONS_START_DATE)]

        # Converting back from PredictionRatio should equals the original column NewCases
        pop_size = cdf['Population'].max()
        curr_total_cases = cdf_prior['ConfirmedCases'].to_numpy()[-1]
        ratios = cdf['PredictionRatio'].to_numpy()
        prev_new_cases = cdf_prior['NewCases'].to_numpy()
        new_cases_reconstructed = oxford_data.convert_prediction_ratios_to_new_cases(
            ratios,
            7,
            prev_new_cases,
            curr_total_cases,
            pop_size,
        )
        original_new_cases = cdf['NewCases'].to_numpy()
        original_smooth_new_cases = cdf['SmoothNewCases'].to_numpy()
        test_cases.append((
            test_country, ratios, new_cases_reconstructed, original_new_cases, original_smooth_new_cases,
        ))

    # Initialize the figure.
    n_cols = 7
    n_rows = 1 + len(test_cases)
    fig = plt.figure(figsize=(3 * n_cols, 2 * n_rows))

    for row, (label, gt_prediction_ratio, computed_new_cases, gt_new_cases, gt_smooth_cases) in enumerate(test_cases):
        label_titles = row == 0
        plot_i = ((row + 1) * n_cols) + 1

        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.axis('off')
        ax.text(0.1, 0.5, label)
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.plot(gt_prediction_ratio, color="purple")
        if label_titles:
            plt.title("Actual PredictionRatio")
        ax = ax_output = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        ax.plot(computed_new_cases, color="blue")
        if label_titles:
            plt.title("Computed NewCases")
        if gt_new_cases is not None:
            ax = plt.subplot(n_rows, n_cols, plot_i, sharey=ax_output); plot_i += 1
            ax.plot(gt_new_cases, color='blue')
            if label_titles:
                plt.title("Actual NewCases")
            ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
            ax.plot(gt_new_cases - computed_new_cases, color='red')
            if label_titles:
                plt.title("Residuals (Actual - Computed)")
        plot_i = ((row + 1) * n_cols) + n_cols - 1 # final-1 column of the row
        ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
        dma = running_mean_convolve(computed_new_cases, 7)
        ax.plot(dma, color="black")
        if label_titles:
            plt.title("Computed SmoothNewCases")
        if gt_smooth_cases is not None:
            ax = plt.subplot(n_rows, n_cols, plot_i); plot_i += 1
            ax.plot(gt_smooth_cases[6:], color="black")
            if label_titles:
                plt.title("Actual SmoothNewCases")

    figure_path = figures_dir / f"04_backward_pass.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(figure_path)


def main():
    # test_backward_pass()
    test_convert_smooth_new_cases_per_100k()
    # test_convert_prediction_ratios()
    # test_plot_case_columns()


if __name__ == '__main__':
    main()


