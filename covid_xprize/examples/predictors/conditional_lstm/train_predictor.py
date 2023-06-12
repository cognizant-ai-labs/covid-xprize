# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import io
import logging
import os
import warnings
from datetime import datetime

import h5py
import keras.backend as K
import numpy as np
import pandas as pd
import pytz
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Lambda
from keras.models import Model
from keras.constraints import Constraint

from covid_xprize.examples.predictors.conditional_lstm.conditional_lstm_model import construct_conditional_lstm_model
from covid_xprize.oxford_data import most_affected_countries, create_country_samples, threshold_min_cases


def construct_model(nb_context: int, nb_action: int, lstm_size: int = 32, nb_lookback_days: int = 21) -> Model:
    return construct_conditional_lstm_model(nb_context=nb_context,
                                            nb_action=nb_action,
                                            lstm_size=lstm_size,
                                            nb_lookback_days=nb_lookback_days)


def train_predictor(training_data: pd.DataFrame,
                    nb_lookback_days: int,
                    nb_training_days: int,
                    nb_test_days: int,
                    nb_training_geos: int,
                    nb_testing_geos: int,
                    nb_trials: int,
                    nb_epochs: int,
                    lstm_size: int,
                    verbose = False) -> tuple[Model, pd.DataFrame]:
    """Trains a prediction model using the given hyperparameter arguments. 
    :param nb_lookback_days: This option is not fully implemented yet. Completing implementation
        will at least require changes to create_country_samples in forecast.py.
        Number of past days included in the context of each sample fed in to
        the predictor. Increasing this can enable the model to better
        disambiguate similar contexts, but could also increase the chance of
        spurious correlations.
    :param nb_training_days: Number of past days to use for training. Used to reduce detrimental
        influence of old data that is no longer relevant comapred to new data.
        Set to None to use all available data.
    :param nb_test_days: Number of days to use for testing and selecting the best model.
        Increasing this improves selection but reduces training data.
    :param nb_training_geos: Number of geos to use for training. Increasing this can potentially
        reduce bias in the data, but could also increase the noise level. If set to None, all geos will be used.
    :param nb_testing_geos: Number of geos to use for testing. This can be different than number
        of training geos in order to do fair comparisons across predictors
        with different nb_training_geos. If set to None, all geos will be used.
    :param nb_trials: Number of trials that will be run from which the best model will be
        selected. This is also a useful option for testing purposes.
    :param nb_epochs: Maximum number of training epochs for the predictor.
        E.g., set to 1 for fast testing.
    :param context_column: Which column in the data df to use as context and outcome.
    :param arch: Which predictor architecture to use.
        Current options are 'conditional' and 'independent'.
    :param verbose: Verbosity level for model.fit() when training the predictor.
    :returns: (best_model, results_df)
    """
    df = training_data
    context_column = 'SmoothNewCasesPer100K'

    # Only look at data with # cases above a minimum.
    df = threshold_min_cases(df)

    # Create data set for training
    if nb_training_geos == None: # Use all countries
        nb_training_geos = len(df.GeoID.unique())
    if nb_testing_geos == None: # Use all countries
        nb_testing_geos = len(df.GeoID.unique())

    train_countries = most_affected_countries(df, nb_training_geos, nb_lookback_days)
    test_countries = most_affected_countries(df, nb_testing_geos, nb_lookback_days)

    # Create numpy arrays for Keras for each country
    if nb_training_geos > nb_testing_geos:
        country_samples = create_country_samples(df, train_countries, context_column, nb_test_days)
    else:
        country_samples = create_country_samples(df, test_countries, context_column, nb_test_days)

    if nb_training_days is not None:
        # Trim training data to nb_training_days days
        for c in country_samples:
            country_samples[c]['X_train_context'] = country_samples[c]['X_train_context'][-nb_training_days:]
            country_samples[c]['X_train_action'] = country_samples[c]['X_train_action'][-nb_training_days:]
            country_samples[c]['y_train'] = country_samples[c]['y_train'][-nb_training_days:]

    # Aggregate data for training
    all_X_context_list = [country_samples[c]['X_train_context'] for c in train_countries]
    all_X_action_list = [country_samples[c]['X_train_action'] for c in train_countries]
    all_y_list = [country_samples[c]['y_train'] for c in train_countries]
    X_context = np.concatenate(all_X_context_list)
    X_action = np.concatenate(all_X_action_list)
    y = np.concatenate(all_y_list)

    # Aggregate data for testing only on top countries
    test_all_X_context_list = [country_samples[c]['X_test_context'] for c in test_countries]
    test_all_X_action_list = [country_samples[c]['X_test_action'] for c in test_countries]
    test_all_y_list = [country_samples[c]['y_test'] for c in test_countries]
    test_X_context = np.concatenate(test_all_X_context_list)
    test_X_action = np.concatenate(test_all_X_action_list)
    test_y = np.concatenate(test_all_y_list)

    # Run full training several times to find best model
    # and gather data for setting acceptance threshold
    models = []
    train_losses = []
    val_losses = []
    test_losses = []
    for t in range(nb_trials):
        # Shuffle data to create new train/val split
        X_context, X_action, y = _permute_data(X_context, X_action, y, seed=t)

        # Construct model
        nb_context = X_context.shape[-1]
        nb_action = X_action.shape[-1]
        model = construct_model(nb_context=nb_context,
                                nb_action=nb_action,
                                lstm_size=lstm_size,
                                nb_lookback_days=nb_lookback_days)

        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        # Train model
        history = _train_model(model, X_context, X_action, y, epochs=nb_epochs, verbose=verbose)

        # Record training results
        top_epoch = np.argmin(history.history['val_loss'])
        train_loss = history.history['loss'][top_epoch]
        val_loss = history.history['val_loss'][top_epoch]
        test_loss = model.evaluate([test_X_context, test_X_action], [test_y])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        models.append(model)
        print('Train Loss:', train_loss)
        print('Val Loss:', val_loss)
        print('Test Loss:', test_loss)

    # Gather test info
    country_indeps = []
    country_predss = []
    for model in models:
        country_indep, country_preds = _lstm_get_test_rollouts( model,
                                                                df,
                                                                test_countries,
                                                                country_samples,
                                                                context_column)
        country_indeps.append(country_indep)
        country_predss.append(country_preds)

    # Compute daily smooth cases per 100K mae
    test_case_maes = []
    for m in range(len(models)):
        total_loss = 0.
        for c in test_countries:
            true_cases = np.array(df[df.GeoID == c].SmoothNewCasesPer100K)[-nb_test_days:]
            pred_cases = country_predss[m][c][-nb_test_days:]
            if true_cases.shape != pred_cases.shape: # Insufficient data
                continue
            total_loss += np.mean(np.abs(true_cases - pred_cases))
        total_loss /= len(test_countries)
        test_case_maes.append(total_loss)

    # Save metrics from training
    results_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'test_loss': test_losses,
        'test_case_mae': test_case_maes})

    # Select best model
    print("Best test case mae:", np.min(test_case_maes))
    best_model = models[np.argmin(test_case_maes)]
    return best_model, results_df


# Shuffling data prior to train/val split
def _permute_data(X_context, X_action, y, seed=301):
    np.random.seed(seed)
    p = np.random.permutation(y.shape[0])
    X_context = X_context[p]
    X_action = X_action[p]
    y = y[p]
    return X_context, X_action, y


# Train model
def _train_model(model, X_context, X_action, y, epochs=1, batch_size=128, verbose=0):
    early_stopping = EarlyStopping(patience=5,
                                   restore_best_weights=True)
    history = model.fit([X_context, X_action], [y],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping],
                        verbose=verbose)
    return history


# Functions for computing test metrics
def _lstm_roll_out_predictions(model, initial_context_input, initial_action_input, future_action_sequence):
    nb_test_days = future_action_sequence.shape[0]
    pred_output = np.zeros(nb_test_days)
    context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
    action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
    for d in range(nb_test_days):
        action_input[:, :-1] = action_input[:, 1:]
        action_input[:, -1] = future_action_sequence[d]
        pred = model.predict([context_input, action_input])
        pred_output[d] = pred
        context_input[:, :-1] = context_input[:, 1:]
        context_input[:, -1] = pred
    return pred_output


def _lstm_get_test_rollouts(model, df, top_countries, country_samples, context_column):
    country_indep = {}
    country_preds = {}
    for c in top_countries:
        X_test_context = country_samples[c]['X_test_context']
        X_test_action = country_samples[c]['X_test_action']
        country_indep[c] = model.predict([X_test_context, X_test_action])

        initial_context_input = country_samples[c]['X_test_context'][0]
        initial_action_input = country_samples[c]['X_test_action'][0]
        y_test = country_samples[c]['y_test']

        nb_test_days = y_test.shape[0]
        nb_actions = initial_action_input.shape[-1]

        future_action_sequence = np.zeros((nb_test_days, nb_actions))
        future_action_sequence[:nb_test_days] = country_samples[c]['X_test_action'][:, -1, :]
        current_action = country_samples[c]['X_test_action'][:, -1, :][-1]
        preds = _lstm_roll_out_predictions(model,
                                           initial_context_input,
                                           initial_action_input,
                                           future_action_sequence)
        country_preds[c] = preds

    return country_indep, country_preds
