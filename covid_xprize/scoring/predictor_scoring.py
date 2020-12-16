# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

from covid_xprize.examples.predictors.lstm.xprize_predictor import XPrizePredictor


def add_predictor_performance_columns(ranking_df):
        """
        Compute performance measures across predictors and add the results to the ranking_df.

        The `ranking_df` argument must already contain values for the following columns:
        - PredictorName
        - GeoID
        - CountryName
        - RegionName
        - Population
        - Date
        - ActualDailyNewCases
        - PredictedDailyNewCases
        - ActualDailyNewCases7DMA
        - PredictedDailyNewCases7DMA

        The function returns the same ranking_df with the following columns added:
        - DiffDaily
        - Diff7DMA
        - CumulDiff7DMA
        - Cumul-7DMA-MAE-per-100K
        - PredictorRank
        """

        # Add MAE columns
        ranking_df['DiffDaily'] = (ranking_df["ActualDailyNewCases"] -
                                   ranking_df["PredictedDailyNewCases"]).abs()
        ranking_df['Diff7DMA'] = (ranking_df["ActualDailyNewCases7DMA"] -
                                  ranking_df["PredictedDailyNewCases7DMA"]).abs()

        # Compute the cumulative sum of 7DMA errors
        ranking_df['CumulDiff7DMA'] = ranking_df.groupby(["GeoID",
                                                          "PredictorName"])['Diff7DMA'].cumsum()

        # Normalize CumulDiff7DMA by geo population size
        ranking_df['Cumul-7DMA-MAE-per-100K'] = ranking_df['CumulDiff7DMA'] / \
                                                (ranking_df['Population'] / 100000.)

        # Add rank column. I.e., rank of this predictor for this geo on this day.
        # In order to detect ties, CumulDiff7DMA is first rounded to the nearest case.
        # That is, if two predictors have the same rounded CumulDiff7DMA then neither
        # is considered better than the other. Rounding here avoids floating point
        # equality errors when comparing the CumulDiff7DMA of predictors that have
        # predicted the exact same number of daily cases.
        ranking_df['PredictorRank'] = ranking_df.round().groupby(
            ["GeoID", "Date"])['CumulDiff7DMA'].rank(method='average')

        # Sort by 7 days moving average mae per 100K
        ranking_df.sort_values(by=["CountryName",
                                   "RegionName",
                                   "Date",
                                   "Cumul-7DMA-MAE-per-100K"],
                               inplace=True)

        return ranking_df


def add_population_column(df):
    """
    Add population column to df in order to compute performance per 100K of population.
    """

    pop_df = XPrizePredictor._load_additional_context_df()
    return df.merge(pop_df[['GeoID', 'Population']], on=['GeoID'], how='left', suffixes=('', '_y'))

