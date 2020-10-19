# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

#
# Example script for training neat-based prescriptors
# Uses neat-python: pip install neat-python
#

from copy import deepcopy

import neat
import numpy as np
import pandas as pd

from lstm.xprize_predictor import XPrizePredictor

from utils import CASES_COL
from utils import PRED_CASES_COL
from utils import HIST_DATA_FILE_PATH
from utils import IP_COLS
from utils import IP_MAX_VALUES
from utils import TMP_PRESCRIPTION_FILE
from utils import get_predictions
from utils import prepare_df


# Cutoff date for training data
CUTOFF_DATE = '2020-07-31'

# Range of days the prescriptors will be evaluated on.
# To save time during training, this range may be significantly
# shorter than the maximum days a prescriptor can be evaluated on.
EVAL_START_DATE = '2020-08-01'
EVAL_END_DATE = '2020-08-02'

# Number of days the prescriptors will look at in the past.
# Larger values here may make convergence slower, but give
# prescriptors more context. The number of inputs of each neat
# network will be NB_LOOKBACK_DAYS * (IP_COLS + 1).
NB_LOOKBACK_DAYS = 14

# Number of countries to use for training. Again, lower numbers
# here will make training faster, since there will be fewer
# input variables, but could potentially miss out on useful info.
NB_EVAL_COUNTRIES = 10


# Load historical data
print("Reading historical data...")
df = pd.read_csv(HIST_DATA_FILE_PATH,
              parse_dates=['Date'],
              encoding="ISO-8859-1",
              error_bad_lines=False)

# Restrict it to dates before the training cutoff
cutoff_date = pd.to_datetime(CUTOFF_DATE, format='%Y-%m-%d')
df = df[df['Date'] <= cutoff_date]

# Preprocess df, e.g., add NewCases and GeoID columns
print("Processing historical data...")
df = prepare_df(df)

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

        # Make prescriptions one day at a time, feeding resulting
        # predictions from the predictor back into the prescriptor.
        for date in pd.date_range(eval_start_date, eval_end_date):
            date_str = date.strftime("%Y-%m-%d")

            # Prescribe for each geo
            for geo in eval_geos:

                # Prepare input data. Here we use log to place cases
                # on a reasonable scale; many other approaches are possible.
                X_cases = np.log(past_cases[geo][-NB_LOOKBACK_DAYS:] + 1)
                X_ips = past_ips[geo][-NB_LOOKBACK_DAYS:]
                X = np.concatenate([X_cases.flatten(),
                                    X_ips.flatten()])

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

            # Save prescriptions so far to file
            pres_df = pd.DataFrame(df_dict)
            pres_df.to_csv(TMP_PRESCRIPTION_FILE)

            # Make prediction given prescription for all countries
            pred_df = get_predictions(EVAL_START_DATE, date_str, TMP_PRESCRIPTION_FILE)

            # Update past data with new day of prescriptions and predictions
            pres_df['GeoID'] = pres_df['CountryName'] + '__' + pres_df['RegionName'].astype(str)
            pred_df['GeoID'] = pred_df['CountryName'] + '__' + pred_df['RegionName'].astype(str)
            new_pres_df = pres_df[pres_df['Date'] == date_str]
            new_pred_df = pred_df[pred_df['Date'] == date_str]
            for geo in eval_geos:
                geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                for ip_col in IP_COLS:
                    eval_past_ips[geo] = np.append(eval_past_ips[geo],
                                                   geo_pres[ip_col].values[0])
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

        # Computing stringency in this way assumes all ip's are weighted equally.
        # In general, ip's may be weighted according to their cost. Such a weighting
        # could be applied here to compute stringency as a weighted sum instead of
        # the simple mean.
        stringency = pres_df[IP_COLS].mean().mean()

        genome.fitness = -(new_cases * stringency)

        print('Evaluated Genome', genome_id)
        print('New cases:', new_cases)
        print('Stringency:', stringency)
        print('Fitness:', genome.fitness)


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-prescriptor')

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

