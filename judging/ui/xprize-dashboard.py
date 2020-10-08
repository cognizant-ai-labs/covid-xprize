import logging
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import s3fs
from dash.dependencies import Input, Output
from dash_table import DataTable
from dash_table.Format import Format

from judging.common import load_dataset
from judging.common import Constants

# Path where this script lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Access to web app and server
app = dash.Dash(__name__)
app.scripts.config.serve_locally = True
server = app.server  # underlying Flask server

# Set up logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('xprize-dashboard')

# For bootstrapping visual components until we've calculated data
EMPTY_FIGURE = go.Figure()

ALL_GEO = "All"
DEFAULT_GEO = ALL_GEO

NUM_PREV_DAYS_TO_INCLUDE = 6
WINDOW_SIZE = 7

# Wrapper object for accessing S3
FS = s3fs.S3FileSystem()

# Continents and countries
continents_df = pd.read_csv(
    os.path.join(ROOT_DIR, 'data/continents_countries.csv'),
    comment='#',
    usecols=['Continent_Name', 'Country_Name']
)

def _get_ranking_df():
    predictions_dir = f's3://{Constants.S3_BUCKET}/predictions'

    # Get latest rankings
    rankings = FS.ls(predictions_dir, refresh=True)
    rankings.sort(reverse=True)

    rankings_date = rankings[0].rsplit('/', 1)[1]

    s3_rankings_path = f'{predictions_dir}/{rankings_date}/rankings/ranking.csv'
    ranking_df = pd.read_csv(s3_rankings_path, parse_dates=['Date'], encoding="ISO-8859-1")

    ranking_df_with_continents = ranking_df.merge(
        continents_df, how='inner', left_on=['CountryName'], right_on=['Country_Name'], copy=False)

    return ranking_df_with_continents

# TODO: should we be computing the ground truth here in the UI? Or calculating it and persisting it in the Ranking
# script?
def _get_actual_cases(df, start_date, end_date):
    # 1 day earlier to compute the daily diff
    start_date_for_diff = start_date - pd.offsets.Day(WINDOW_SIZE)
    actual_df = df[["CountryName", "RegionName", "Date", "ConfirmedCases"]]
    # Filter out the data set to include all the data needed to compute the diff
    actual_df = actual_df[(actual_df.Date >= start_date_for_diff) & (actual_df.Date <= end_date)]
    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    actual_df["GeoID"] = np.where(actual_df["RegionName"] == '',
                                  actual_df["CountryName"],
                                  actual_df["CountryName"] + ' / ' + actual_df["RegionName"])
    actual_df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the diff
    actual_df["ActualDailyNewCases"] = actual_df.groupby("GeoID")["ConfirmedCases"].diff().fillna(0)
    # Compute the 7 day moving average
    actual_df["ActualDailyNewCases7DMA"] = actual_df \
        .groupby("GeoID")['ActualDailyNewCases'] \
        .rolling(WINDOW_SIZE, center=False) \
        .mean() \
        .reset_index(0, drop=True)

    # Add in continent info
    actual_df_with_continents = actual_df.merge(
        continents_df, how='inner', left_on=['CountryName'], right_on=['Country_Name'], copy=False)
    return actual_df_with_continents


@app.callback(
    [
        Output('continent', 'value'),
        Output('country', 'value'),
        Output('region', 'value')
    ],
    [
        Input('reset', 'n_clicks')
    ]
)
def reset_button(_):
    return DEFAULT_GEO, DEFAULT_GEO, DEFAULT_GEO


@app.callback(
    [
        Output('overall_ranking', 'data'),
        Output('predictions', 'figure'),
        Output('rankings', 'figure'),
        Output('country', 'options'),
        Output('region', 'options')
    ],
    [
        Input('continent', 'value'),
        Input('country', 'value'),
        Input('region', 'value')
    ]
)
def update_figures(selected_continent, selected_country, selected_region):
    continent_to_use = selected_continent or DEFAULT_GEO
    country_to_use = selected_country or DEFAULT_GEO
    region_to_use = selected_region or DEFAULT_GEO

    latest_df = load_dataset()
    ranking_df = _get_ranking_df()

    # Get start and end date by checking extreme date values in ranking_df
    start_date = ranking_df.Date.min()
    end_date = ranking_df.Date.max()

    ground_truth_df = _get_actual_cases(latest_df, start_date, end_date)

    predictions_fig = go.Figure(layout=dict(title=dict(text=f"{continent_to_use}/{country_to_use}/{region_to_use} "
                                                            f"Daily New Cases 7-day Average ",
                                                       y=0.9,
                                                       x=0.5,
                                                       xanchor='center',
                                                       yanchor='top'
                                                       ),
                                            plot_bgcolor='#f2f2f2',
                                            xaxis_title="Date",
                                            yaxis_title="Daily new cases 7-day average"
                                            ))
    # Format x axis
    predictions_fig.update_xaxes(
        dtick="D1",  # Means 1 day
        tickformat="%d\n%b")

    predictor_names = list(ranking_df.PredictorName.dropna().unique())

    filtered_df = ranking_df.copy()
    filtered_ground_truth_df = ground_truth_df.copy()
    if continent_to_use != ALL_GEO:
        filtered_df = filtered_df[filtered_df.Continent_Name == continent_to_use]
        filtered_ground_truth_df = filtered_ground_truth_df[filtered_ground_truth_df.Continent_Name == continent_to_use]

    if country_to_use != ALL_GEO:
        filtered_df = filtered_df[filtered_df.CountryName == country_to_use]
        filtered_ground_truth_df = filtered_ground_truth_df[filtered_ground_truth_df.CountryName == country_to_use]

    if region_to_use != ALL_GEO:
        filtered_df = filtered_df[filtered_df.RegionName == region_to_use]
        filtered_ground_truth_df = filtered_ground_truth_df[filtered_ground_truth_df.RegionName == region_to_use]

    filtered_df = filtered_df \
            .groupby(["PredictorName", "Date"])[["GeoID", "PredictorName", "PredictedDailyNewCases7DMA", "ActualDailyNewCases7DMA"]] \
            .sum() \
            .sort_values(by=["PredictorName", "Date"]).reset_index()

    # Add 1 trace per predictor, for selected geo area
    for predictor_name in predictor_names:
        predictor_df = filtered_df[filtered_df.PredictorName == predictor_name]
        predictions_fig.add_trace(
            go.Scatter(x=predictor_df.Date,
                       y=predictor_df.PredictedDailyNewCases7DMA,
                       name=predictor_name,
                       visible=(ALL_GEO == DEFAULT_GEO))
        )

    # Add 1 trace for the overall ground truth
    filtered_ground_truth_df = filtered_ground_truth_df[filtered_ground_truth_df.Date >= start_date] \
        .groupby(["Date"])[["GeoID", "ActualDailyNewCases7DMA"]]\
        .sum()\
        .sort_values(by=["Date"])\
        .reset_index()
    predictions_fig.add_trace(
        go.Scatter(
            x=filtered_ground_truth_df.Date,
            y=filtered_ground_truth_df.ActualDailyNewCases7DMA,
            name="Ground Truth",
            visible=(ALL_GEO == DEFAULT_GEO),
            line=dict(color='orange', width=4, dash='dash')
        )
    )

    # rankings / mean errors
    ranking_fig = go.Figure(layout=dict(title=dict(text=f'{DEFAULT_GEO} submission rankings',
                                                   y=0.9,
                                                   x=0.5,
                                                   xanchor='center',
                                                   yanchor='top'
                                                   ),
                                        plot_bgcolor='#f2f2f2',
                                        xaxis_title="Date",
                                        yaxis_title="Cumulative 7DMA error"
                                        ))
    # Format x axis
    ranking_fig.update_xaxes(
        dtick="D1",  # Means 1 day
        tickformat="%d\n%b")

    # Add 1 trace per predictor showing MAE for the selected geo ID
    filtered_df['Diff7DMA'] = (filtered_df["ActualDailyNewCases7DMA"] - filtered_df["PredictedDailyNewCases7DMA"]).abs()
    filtered_df['CumulDiff7DMA'] = filtered_df.groupby(["PredictorName"])['Diff7DMA'].cumsum()
    for predictor_name in predictor_names:
        predictor_df = filtered_df[filtered_df.PredictorName == predictor_name]
        ranking_fig.add_trace(
            go.Scatter(
                x=predictor_df.Date,
                y=predictor_df.CumulDiff7DMA,
                name=predictor_name
            )
        )

    # Figure out regions for selected continent
    # If it's "all continents" (DEFAULT_GEO) then list all countries
    if continent_to_use == DEFAULT_GEO:
        countries_list = [DEFAULT_GEO] + list(ranking_df.CountryName.dropna().sort_values().unique())
    else:
        countries_list = [DEFAULT_GEO] + list(
            ranking_df[ranking_df['Continent_Name'] == continent_to_use]['CountryName'].dropna().sort_values().unique())
    countries_dict = [{'label': c, 'value': c} for c in countries_list]

    if country_to_use == DEFAULT_GEO:
        regions_list = [DEFAULT_GEO] + list(ranking_df.RegionName.dropna().sort_values().unique())
    else:
        regions_list = [DEFAULT_GEO] + list(
            ranking_df[ranking_df['CountryName'] == country_to_use]['RegionName'].dropna().sort_values().unique())

    regions_dict = [{'label': r, 'value': r} for r in regions_list]

    overall_ranking_df = ranking_df[['PredictorName', 'CumulDiff7DMA']] \
        .groupby('PredictorName') \
        .sum() \
        .round({'CumulDiff7DMA': 0}) \
        .sort_values(by='CumulDiff7DMA') \
        .reset_index()

    overall_ranking_df.rename(columns={'PredictorName': 'Team', 'CumulDiff7DMA': 'Score'}, inplace=True)

    # insert rank
    overall_ranking_df.insert(0, 'Rank', range(1, len(overall_ranking_df) + 1))

    return overall_ranking_df.to_dict('rows'), predictions_fig, ranking_fig, countries_dict, regions_dict


def main():
    global app
    app.title = 'Cognizant COVID X-Prize'
    app.css.config.serve_locally = False
    # Don't be afraid of the 3rd party URLs: chriddyp is the author of Dash!
    # These two allow us to dim the screen while loading.
    # See discussion with Dash devs here: https://community.plotly.com/t/dash-loading-states/5687
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'})

    continents = [DEFAULT_GEO] + list(continents_df.Continent_Name.sort_values().unique())
    continents_dict = [{'label': c, 'value': c} for c in continents]
    app.layout = html.Div(
        className="container scalable",
        children=[
            html.H1('Cognizant COVID X-Prize: Current standings'),
            dcc.Store(id='memory'),
            html.Div(
                className="app_main_content",
                children=[
                    html.Div(
                        className="row",
                        children=[
                            html.Div(
                                className="four columns",
                                children=[
                                    html.Div(
                                        children=[
                                            html.P("Continent", style={"color": "#515151", "margin-top": "0px"}),
                                            dcc.Dropdown(
                                                id='continent',
                                                options=continents_dict,
                                                value=DEFAULT_GEO
                                            ),
                                            html.P("Country", style={"color": "#515151", "margin-top": "0px"}),
                                            dcc.Dropdown(
                                                id='country',
                                                options=[{}],
                                                value=DEFAULT_GEO
                                            ),
                                            html.P("Region", style={"color": "#515151", "margin-top": "0px"}),
                                            dcc.Dropdown(
                                                id='region',
                                                options=[{}],
                                                value=DEFAULT_GEO
                                            ),
                                            html.Br(),
                                            html.Button('Reset', id='reset')
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    html.H3('Overall ranking', style={'backgroundColor': 'white'}),
                                    DataTable(id='overall_ranking',
                                              columns=[
                                                  {'name': 'Rank', 'id': 'Rank'},
                                                  {'name': 'Team', 'id': 'Team'},
                                                  {'name': 'Score', 'id': 'Score', 'type': 'numeric',
                                                   'format': Format(group=',')}
                                              ],
                                              style_as_list_view=False,
                                              style_cell={'padding': '5px'},
                                              style_header={
                                                  'backgroundColor': 'white',
                                                  'fontWeight': 'bold'
                                              },
                                              style_cell_conditional=[
                                                  {
                                                      'if': {'column_id': ['Team']},
                                                      'textAlign': 'left'
                                                  }
                                              ],
                                              data=None),
                                    html.Br(),
                                    dcc.Graph(id='predictions', figure=EMPTY_FIGURE),
                                    dcc.Graph(id='rankings', figure=EMPTY_FIGURE)
                                ],
                                className="eight columns",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    app.run_server(host='0.0.0.0', debug=False, port=4058, use_reloader=False, threaded=False)


if __name__ == '__main__':
    main()















