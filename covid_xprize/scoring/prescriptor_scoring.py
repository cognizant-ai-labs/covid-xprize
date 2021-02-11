import time

import pandas as pd

from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor
from covid_xprize.standard_predictor.xprize_predictor import NPI_COLUMNS


def weight_prescriptions_by_cost(pres_df, cost_df):
    """
    Weight prescriptions by their costs.
    """
    weighted_df = pres_df.merge(cost_df, how='outer', on=['CountryName', 'RegionName'], suffixes=('_pres', '_cost'))
    for npi_col in NPI_COLUMNS:
        weighted_df[npi_col] = weighted_df[npi_col + '_pres'] * weighted_df[npi_col + '_cost']
    return weighted_df


def generate_cases_and_stringency_for_prescriptions(start_date, end_date, prescription_file, costs_file, past_ips_file=None):
    start_time = time.time()
    # Load the prescriptions, handling Date and regions
    pres_df = XPrizePredictor.load_original_data(prescription_file)

    # Generate predictions for all prescriptions
    predictor = XPrizePredictor()
    pred_dfs = {}
    for idx in pres_df['PrescriptionIndex'].unique():
        idx_df = pres_df[pres_df['PrescriptionIndex'] == idx]
        idx_df = idx_df.drop(columns='PrescriptionIndex')  # Predictor doesn't need this

        # Prepend past ips if provided. This is used to handle the case when
        # prescriptions start after predictor's dataset ends.
        if past_ips_file:
            past_ips_df = XPrizePredictor.load_original_data(past_ips_file)
            past_ips_df = past_ips_df[past_ips_df['Date'] < start_date]
            idx_df = past_ips_df.append(idx_df)

        # Generate the predictions
        pred_df = predictor.predict_from_df(start_date, end_date, idx_df)
        print(f"Generated predictions for PrescriptionIndex {idx}")
        pred_df['PrescriptionIndex'] = idx
        pred_dfs[idx] = pred_df
    pred_df = pd.concat(list(pred_dfs.values()))

    # Aggregate cases by prescription index and geo
    agg_pred_df = pred_df.groupby(['CountryName',
                                   'RegionName',
                                   'PrescriptionIndex'], dropna=False).mean().reset_index()

    # Load IP cost weights
    cost_df = pd.read_csv(costs_file)

    # Only use costs of geos we've predicted for
    cost_df = cost_df[cost_df.CountryName.isin(agg_pred_df.CountryName) &
                      cost_df.RegionName.isin(agg_pred_df.RegionName)]

    # Apply weights to prescriptions
    pres_df = weight_prescriptions_by_cost(pres_df, cost_df)

    # Aggregate stringency across npis
    pres_df['Stringency'] = pres_df[NPI_COLUMNS].sum(axis=1)

    # Aggregate stringency by prescription index and geo
    agg_pres_df = pres_df.groupby(['CountryName',
                                   'RegionName',
                                   'PrescriptionIndex'], dropna=False).mean().reset_index()

    # Combine stringency and cases into a single df
    df = agg_pres_df.merge(agg_pred_df, how='outer', on=['CountryName',
                                                         'RegionName',
                                                         'PrescriptionIndex'])

    # Only keep columns of interest
    df = df[['CountryName',
             'RegionName',
             'PrescriptionIndex',
             'PredictedDailyNewCases',
             'Stringency']]
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_tring = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Evaluated {len(pred_dfs)} PrescriptionIndex in {elapsed_time_tring} seconds")

    return df, pred_dfs


# Compute domination relationship for each pair of prescriptors for each geo
def compute_domination_df(df):
    country_names = []
    region_names = []
    dominating_names = []
    dominated_names = []
    dominated_idxs = []
    prescriptor_names = sorted(df['PrescriptorName'].unique())
    for country_name in df['CountryName'].unique():
        cdf = df[df['CountryName'] == country_name]
        for region_name in cdf['RegionName'].unique():
            print('Computing domination for', country_name, region_name)
            if pd.isna(region_name):
                rdf = cdf[cdf['RegionName'].isna()]
            else:
                rdf = cdf[cdf['RegionName'] == region_name]
            for name1 in prescriptor_names:
                dominated_prescriptions = set()  # To collect all dominated prescriptions
                # this is a set, so each dominated
                # prescription is only counted once per
                # submission.
                name1_df = rdf[rdf['PrescriptorName'] == name1]
                name1_idxs = sorted(name1_df['PrescriptionIndex'].unique())
                for i in name1_idxs:
                    idf = name1_df[name1_df['PrescriptionIndex'] == i]
                    i_cases = idf['PredictedDailyNewCases'].values[0]
                    i_stringency = idf['Stringency'].values[0]

                    # Compare name1 to all other prescriptions
                    for name2 in prescriptor_names:
                        if name1 != name2:
                            name2_df = rdf[rdf['PrescriptorName'] == name2]
                            name2_idxs = sorted(name2_df['PrescriptionIndex'].unique())
                            for j in name2_idxs:
                                jdf = name2_df[name2_df['PrescriptionIndex'] == j]
                                j_cases = jdf['PredictedDailyNewCases'].values[0]
                                j_stringency = jdf['Stringency'].values[0]
                                if (i_cases < j_cases) and (i_stringency < j_stringency):
                                    dominated_prescriptions.add((name2, j))
                for name2, j in dominated_prescriptions:
                    country_names.append(country_name)
                    region_names.append(region_name)
                    dominating_names.append(name1)
                    dominated_names.append(name2)
                    dominated_idxs.append(j)
    ddf = pd.DataFrame({
        'CountryName': country_names,
        'RegionName': region_names,
        'DominatingName': dominating_names,
        'DominatedName': dominated_names,
        'DominatedIndex': dominated_idxs})
    return ddf


#
# Helpful functions for visualizing the area dominated by a set of solutions.
#
def compute_pareto_set(objective1_list, objective2_list):
    """
    Return objective values for the subset of solutions that
    lie on the pareto front.
    """

    assert len(objective1_list) == len(objective2_list), \
        "Each solution must have a value for each objective."

    n_solutions = len(objective1_list)

    objective1_pareto = []
    objective2_pareto = []
    for i in range(n_solutions):
        is_in_pareto_set = True
        for j in range(n_solutions):
            if (objective1_list[j] < objective1_list[i]) and \
                    (objective2_list[j] < objective2_list[i]):
                is_in_pareto_set = False
        if is_in_pareto_set:
            objective1_pareto.append(objective1_list[i])
            objective2_pareto.append(objective2_list[i])

    return objective1_pareto, objective2_pareto
