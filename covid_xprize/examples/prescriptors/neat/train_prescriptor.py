# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

#
# Example script for training neat-based prescriptors
# Uses neat-python: pip install neat-python
#
import os
from copy import deepcopy

import neat
import numpy as np
import pandas as pd
from pathlib import Path

from covid_xprize.examples.prescriptors.neat.utils import PRED_CASES_COL, prepare_historical_df, CASES_COL, IP_COLS, \
    IP_MAX_VALUES, add_geo_id, get_predictions

# Cutoff date for training data
from covid_xprize.validation.cost_generator import generate_costs

# Path where this script lives
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Config file for running NEAT (expected to reside in same dir as this script)
NEAT_CONFIG_FILE = ROOT_DIR / 'config-prescriptor'

CUTOFF_DATE = '2020-07-31'

# Range of days the prescriptors will be evaluated on.
# To save time during training, this range may be significantly
# shorter than the maximum days a prescriptor can be evaluated on.
EVAL_START_DATE = '2020-08-01'
EVAL_END_DATE = '2020-08-02'

# Number of days the prescriptors will look at in the past.
# Larger values here may make convergence slower, but give
# prescriptors more context. The number of inputs of each neat
# network will be NB_LOOKBACK_DAYS * (IP_COLS + 1) + IP_COLS.
# The '1' is for previous case data, and the final IP_COLS
# is for IP cost information.
NB_LOOKBACK_DAYS = 14

# Number of countries to use for training. Again, lower numbers
# here will make training faster, since there will be fewer
# input variables, but could potentially miss out on useful info.
NB_EVAL_COUNTRIES = 10


# Load historical data with basic preprocessing
print("Loading historical data...")
df = prepare_historical_df()

# Restrict it to dates before the training cutoff
cutoff_date = pd.to_datetime(CUTOFF_DATE, format='%Y-%m-%d')
df = df[df['Date'] <= cutoff_date]

# As a heuristic, use the top NB_EVAL_COUNTRIES w.r.t. ConfirmedCases
# so far as the geos for evaluation.
eval_geos = list(df.groupby('GeoID').max()['ConfirmedCases'].sort_values(
                ascending=False).head(NB_EVAL_COUNTRIES).index)
print("Nets will be evaluated on the following geos:", eval_geos)

# Pull out historical data for all geos
past_cases = {}
past_ips = {}
for geo in eval_geos:
    geo_df = df[df['GeoID'] == geo]
    past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))
    past_ips[geo] = np.array(geo_df[IP_COLS])

# Gather values for scaling network output
ip_max_values_arr = np.array([IP_MAX_VALUES[ip] for ip in IP_COLS])

# Do any additional setup that is constant across evaluations
eval_start_date = pd.to_datetime(EVAL_START_DATE, format='%Y-%m-%d')
eval_end_date = pd.to_datetime(EVAL_END_DATE, format='%Y-%m-%d')


# Function that evaluates the fitness of each prescriptor model
def eval_genomes(genomes, config):

    # Every generation sample a different set of costs per geo,
    # so that over time solutions become robust to different costs.
    cost_df = generate_costs(distribution='uniform')
    cost_df = add_geo_id(cost_df)
    geo_costs = {}
    for geo in eval_geos:
        costs = cost_df[cost_df['GeoID'] == geo]
        cost_arr = np.array(costs[IP_COLS])[0]
        geo_costs[geo] = cost_arr

    # Evaluate each individual
    for genome_id, genome in genomes:

        # Create net from genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Set up dictionary to keep track of prescription
        df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
        for ip_col in IP_COLS:
            df_dict[ip_col] = []

        # Set initial data
        eval_past_cases = deepcopy(past_cases)
        eval_past_ips = deepcopy(past_ips)

        # Compute prescribed stringency incrementally
        stringency = 0.

        # Make prescriptions one day at a time, feeding resulting
        # predictions from the predictor back into the prescriptor.
        for date in pd.date_range(eval_start_date, eval_end_date):
            date_str = date.strftime("%Y-%m-%d")

            # Prescribe for each geo
            for geo in eval_geos:

                # Prepare input data. Here we use log to place cases
                # on a reasonable scale; many other approaches are possible.
                X_cases = np.log(eval_past_cases[geo][-NB_LOOKBACK_DAYS:] + 1)
                X_ips = eval_past_ips[geo][-NB_LOOKBACK_DAYS:]
                X_costs = geo_costs[geo]
                X = np.concatenate([X_cases.flatten(),
                                    X_ips.flatten(),
                                    X_costs])

                # Get prescription
                prescribed_ips = net.activate(X)

                # Map prescription to integer outputs
                prescribed_ips = (prescribed_ips * ip_max_values_arr).round()

                # Add it to prescription dictionary
                country_name, region_name = geo.split('__')
                if region_name == 'nan':
                    region_name = np.nan
                df_dict['CountryName'].append(country_name)
                df_dict['RegionName'].append(region_name)
                df_dict['Date'].append(date_str)
                for ip_col, prescribed_ip in zip(IP_COLS, prescribed_ips):
                    df_dict[ip_col].append(prescribed_ip)

                # Update stringency. This calculation could include division by
                # the number of IPs and/or number of geos, but that would have
                # no effect on the ordering of candidate solutions.
                stringency += np.sum(geo_costs[geo] * prescribed_ips)

            # Create dataframe from prescriptions.
            pres_df = pd.DataFrame(df_dict)

            # Make prediction given prescription for all countries
            pred_df = get_predictions(EVAL_START_DATE, date_str, pres_df)

            # Update past data with new day of prescriptions and predictions
            pres_df['GeoID'] = pres_df['CountryName'] + '__' + pres_df['RegionName'].astype(str)
            pred_df['RegionName'] = pred_df['RegionName'].fillna("")
            pred_df['GeoID'] = pred_df['CountryName'] + '__' + pred_df['RegionName'].astype(str)
            new_pres_df = pres_df[pres_df['Date'] == date_str]
            new_pred_df = pred_df[pred_df['Date'] == date_str]
            for geo in eval_geos:
                geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]

                # Append array of prescriptions
                pres_arr = np.array([geo_pres[ip_col].values[0] for ip_col in IP_COLS]).reshape(1,-1)
                eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

                # Append predicted cases
                eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                 geo_pred[PRED_CASES_COL].values[0])

        # Compute fitness. There are many possibilities for computing fitness and ranking
        # candidates. Here we choose to minimize the product of ip stringency and predicted
        # cases. This product captures the area of the 2D objective space that dominates
        # the candidate. We minimize it by including a negation. To place the fitness on
        # a reasonable scale, we take means over all geos and days. Note that this fitness
        # function can lead directly to the degenerate solution of all ips 0, i.e.,
        # stringency zero. To achieve more interesting behavior, a different fitness
        # function may be required.
        new_cases = pred_df[PRED_CASES_COL].mean().mean()
        genome.fitness = -(new_cases * stringency)

        print('Evaluated Genome', genome_id)
        print('New cases:', new_cases)
        print('Stringency:', stringency)
        print('Fitness:', genome.fitness)


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     NEAT_CONFIG_FILE)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(show_species_detail=True))

# Add statistics reporter to provide extra info about training progress.
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Add checkpointer to save population every generation and every 10 minutes.
p.add_reporter(neat.Checkpointer(generation_interval=1,
                                 time_interval_seconds=600,
                                 filename_prefix='neat-checkpoint-'))

# Run until a solution is found. Since a "solution" as defined in our config
# would have 0 fitness, this will run indefinitely and require manual stopping,
# unless evolution finds the solution that uses 0 for all ips. A different
# value can be placed in the config for automatic stopping at other thresholds.
winner = p.run(eval_genomes)

# At any time during evolution, we can inspect the latest saved checkpoint
# neat-checkpoint-* to see how well it is doing.

