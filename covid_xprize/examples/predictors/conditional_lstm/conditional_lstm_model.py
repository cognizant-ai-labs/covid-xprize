"""
This model implements a generalization of the positivity constraint of the effects of NPIS
by cascading the context LSTM into the action LSTM instead of running them in parallel.

This version places the LSTM states of the context LSTM into the action LSTM. Thus,
the action LSTM is conditioned on the result of the context LSTM.
"""

import numpy as np

from keras.initializers import TruncatedNormal
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import SpatialDropout1D
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import Constraint
from keras import backend as K

class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)

# Use this at the NPI input so that higher NPIs yield lower cases.
def _invert_npis(x):
    return 1. - (x + 1.) / 5.

# Scale the context and outcome so they are in a reasonable range at initialization.
# This does not affect the theoretical expressivity of the model.
def _scale_context(x):
    return x / 100.

def _scale_outcome(x):
    return x * 100.

def construct_conditional_lstm_model(nb_context: int,
                                     nb_action: int,
                                     lstm_size: int = 16,
                                     nb_lookback_days: int = 21,
                                     weight_decay: float = 0.,
                                     input_dropout: float = 0.,
                                     recurrent_dropout: float = 0.) -> Model:

    # Create context encoder
    context_input = Input(shape=(nb_lookback_days, nb_context),
                          name='context_input')
    x = Lambda(_scale_context)(context_input)
    encoded_outputs, state_h, state_c = LSTM(lstm_size,
                                             return_state=True,
                                             kernel_regularizer=l2(weight_decay),
                                             recurrent_regularizer=l2(weight_decay),
                                             bias_regularizer=l2(weight_decay),
                                             dropout=input_dropout,
                                             recurrent_dropout=recurrent_dropout,
                                             activation='softsign',
                                             name='context_lstm')(x)

    # Create projections of context lstm state to feed to action lstm
    projected_state_h = Dense(lstm_size,
                              activation='relu',
                              kernel_regularizer=l2(weight_decay),
                              bias_regularizer = l2(weight_decay),
                              name='project_h_dense')(state_h)
    projected_state_c = Dense(lstm_size,
                              activation='relu',
                              kernel_regularizer=l2(weight_decay),
                              bias_regularizer = l2(weight_decay),
                              name='project_c_dense')(state_c)
    projected_states = [projected_state_h, projected_state_c]

    # Create action encoder
    action_input = Input(shape=(nb_lookback_days, nb_action),
                         name='action_input')

    inverted_action_input = Lambda(_invert_npis, name='npi_inversion')(action_input)

    x = LSTM(units=lstm_size,
             kernel_constraint=Positive(),
             kernel_initializer=TruncatedNormal(stddev=0.0001),
             recurrent_constraint=Positive(),
             bias_constraint=Positive(),
             kernel_regularizer=l2(weight_decay),
             recurrent_regularizer=l2(weight_decay),
             bias_regularizer = l2(weight_decay),
             dropout=input_dropout,
             recurrent_dropout=recurrent_dropout,
             activation='softsign',
             name='action_lstm')(inverted_action_input, initial_state=projected_states)

    # Produce output
    model_output = Dense(units=1,
                         activation='relu',
                         kernel_constraint=Positive(),
                         kernel_regularizer=l2(weight_decay),
                         bias_regularizer=l2(weight_decay),
                         name='action_dense')(x)

    scaled_output = Lambda(_scale_outcome)(model_output)

    # Create prediction model
    model = Model(inputs=[context_input, action_input],
                  outputs=[scaled_output])

    return model



if __name__ == '__main__':
    model = construct_conditional_lstm_model(np.zeros((21, 1)),
                                             np.zeros((21, 12)),
                                             lstm_size=32,
                                             nb_lookback_days=21)
    model.summary()
