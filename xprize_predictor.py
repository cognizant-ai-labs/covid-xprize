import keras.backend as K
from keras.constraints import Constraint


class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)


class XPrizePredictor(object):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self):
        pass

    def submission_predict(self, prediction_day, npis):
        """
        Makes a prediction of daily confirmed cases for a given day.
        :param prediction_day: the day for which to make the prediction
        :param npis: the actual npis between the submission date (excluded) and the prediction_day
        :return: a Pandas DataFrame containing the prediction [`CountryName`, `Date`, `NewCases]
        """
        return 0

    def simple_roll_out(self,
                        predictor,
                        start_date,
                        nb_pred_days,
                        countries,
                        country_samples,
                        df,
                        prescriptor=None,
                        prescription_freq=1,
                        historical_npi_df=None):
        # Prepare the output
        forecast = {"CountryName": [],
                    "Date": [],
                    "ConfirmedCases": [],
                    "NewCases": [],
                    "Forecast": [],
                    "CurrentNpisConfirmedCases": [],
                    "CurrentNpisNewCases": []}

        for c in countries:
            cdf = df[df.CountryName == c]
            cdf = cdf[cdf.ConfirmedCases.notnull()]
            # Get predictions
            if start_date is None:
                forecast_date = max(cdf.Date.unique()) + np.timedelta64(1, 'D')
            else:
                forecast_date = start_date
            # If forecast_date is 4/23
            # df last row is 4/23
            # country_samples last sample is 4/23, which means context from 4/1 to 4/22 and we try to predict
            # CaseRatio for 4/23 and following nb_pred_days
            initial_context_input = country_samples[c]['X_test_context'][-1]
            initial_action_input = country_samples[c]['X_test_action'][-1]
            # y_test = country_samples[c]['y_test']
            # nb_test_days = y_test.shape[0]
            nb_actions = initial_action_input.shape[-1]

            # Predictions with current npis
            future_action_sequence = np.zeros((nb_pred_days, nb_actions))
            current_action = initial_action_input[-1]
            future_action_sequence[:] = current_action

            # Get the predictions if current npi remain in place in the future
            current_npis_preds, _ = MultiplicativeModelEvaluator.roll_out_predictions(
                predictor,
                initial_context_input,
                initial_action_input,
                future_action_sequence,
                prescriptor=None,
                npi_scaler=None,
                prescription_freq=prescription_freq)

            # Get the predictions and the prescribed future actions
            preds, prescribed_actions = MultiplicativeModelEvaluator.roll_out_predictions(predictor,
                                                                                          initial_context_input,
                                                                                          initial_action_input,
                                                                                          future_action_sequence,
                                                                                          prescriptor,
                                                                                          npi_scaler,
                                                                                          prescription_freq)

            # Gather info to convert to total cases
            prev_confirmed_cases = np.array(cdf.ConfirmedCases)
            prev_new_cases = np.array(cdf.NewCases)
            initial_total_cases = prev_confirmed_cases[-1]
            pop_size = np.array(cdf.Population)[-1]  # Population size doesn't change over time

            # Compute predictor's forecast
            pred_total_cases, pred_new_cases = MultiplicativeModelEvaluator.convert_ratios_to_total_cases(
                preds,
                self.window_size,
                prev_new_cases,
                initial_total_cases,
                pop_size)
            # Smooth out pred_new_cases
            # If window size is 7, take the previous 6 new cases so we start doing a 7 day moving average for the first
            # pred new cases
            temp_pred_new_cases = list(prev_new_cases[-(self.window_size-1):]) + pred_new_cases
            smooth_pred_new_cases = MultiplicativeModelEvaluator.smooth_case_list(
                temp_pred_new_cases, self.window_size)
            # Get rid of the first window_size - 1 NaN values where
            # there was not enough data to compute a moving average
            pred_new_cases = smooth_pred_new_cases[self.window_size-1:]

            # Convert to total cases for current NPIs
            current_npis_prev_confirmed_cases = np.array(cdf.CurrentNpisConfirmedCases)
            current_npis_prev_new_cases = np.array(cdf.CurrentNpisNewCases)
            current_npis_initial_total_cases = current_npis_prev_confirmed_cases[-1]

            current_npis_pred_total_cases, current_npis_pred_new_cases = \
                MultiplicativeModelEvaluator.convert_ratios_to_total_cases(current_npis_preds,
                                                                           self.window_size,
                                                                           current_npis_prev_new_cases,
                                                                           current_npis_initial_total_cases,
                                                                           pop_size)

            # Append forecast data to results to return
            for i, pred in enumerate(preds):
                forecast["CountryName"].append(c)
                current_date = forecast_date + np.timedelta64(i, 'D')
                forecast["Date"].append(current_date)
                forecast["ConfirmedCases"].append(pred_total_cases[i])
                forecast["NewCases"].append(pred_new_cases[i])
                for npi, npi_value in zip(self.action_names, prescribed_actions[i]):
                    forecast[npi].append(npi_value)
                forecast["Forecast"].append(True)
                forecast["CurrentNpisConfirmedCases"].append(round(current_npis_pred_total_cases[i]))
                forecast["CurrentNpisNewCases"].append(round(current_npis_pred_new_cases[i]))

        forecast_df = pd.DataFrame.from_dict(forecast)
        return forecast_df
