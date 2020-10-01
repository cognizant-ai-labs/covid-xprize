import logging
from datetime import date

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# Dash app
import s3fs
from dash.dependencies import Input, Output

from common.common_routines import load_dataset
from common.constants import Constants

# Access to web app and server
app = dash.Dash(__name__)
server = app.server  # underlying Flask server

# Set up logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('xprize-dashboard')

# For bootstrapping visual components until we've calculated data
EMPTY_FIGURE = go.Figure()

ALL_GEO = "Overall"
DEFAULT_GEO = ALL_GEO

NUM_PREV_DAYS_TO_INCLUDE = 6
WINDOW_SIZE = 7

# Wrapper object for accessing S3
FS = s3fs.S3FileSystem()


def _get_ranking_df():
    today_date = date.today().strftime("%Y_%m_%d")
    s3_rankings_path = f's3://{Constants.S3_BUCKET}/predictions/{today_date}/rankings/ranking.csv'
    ranking_df = pd.read_csv(s3_rankings_path, parse_dates=['Date'], encoding="ISO-8859-1")
    return ranking_df


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
    #     actual_df['GeoID'] = actual_df['CountryName'] + '__' + actual_df['RegionName'].astype(str)
    actual_df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the diff
    actual_df["ActualDailyNewCases"] = actual_df.groupby("GeoID")["ConfirmedCases"].diff().fillna(0)
    # Compute the 7 day moving average
    actual_df["ActualDailyNewCases7DMA"] = actual_df.groupby(
        "GeoID")['ActualDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)
    #     # Return only the data between start_date and end_date
    #     actual_df = actual_df[(actual_df.Date >= start_date) & (actual_df.Date <= end_date)]
    return actual_df


@app.callback(
    [
        Output('rankings', 'figure')
    ],
    [
        Input('selected_geo_id', 'value')
    ]
)
def update_figures(selected_geo_id):
    latest_df = load_dataset()

    # TODO: make these params
    start_date = pd.to_datetime('2020-08-01', format='%Y-%m-%d')
    end_date = pd.to_datetime('2020-08-04', format='%Y-%m-%d')

    actual_df = _get_actual_cases(latest_df, start_date, end_date)

    ranking_df = _get_ranking_df()

    fig = go.Figure(layout=dict(title=dict(text=f"{DEFAULT_GEO} Daily New Cases 7-day Average ",
                                           y=0.9,
                                           x=0.5,
                                           xanchor='center',
                                           yanchor='top'
                                           ),
                                plot_bgcolor='#f2f2f2',
                                xaxis_title="Date",
                                yaxis_title="Daily new cases 7-day average"
                                ))

    # Keep track of trace visibility by geo ID name
    geoid_plot_names = []

    all_df = ranking_df.groupby(["PredictorName", "Date"])[["GeoID", "PredictorName", "PredictedDailyNewCases7DMA"]].sum().sort_values(by=["PredictorName", "Date"]).reset_index()
    predictor_names = list(ranking_df.PredictorName.dropna().unique())
    geoid_names = list(ranking_df.GeoID.unique())

    # Add 1 trace per predictor, for all geos
    for predictor_name in predictor_names:
        all_geo_df = all_df[all_df.PredictorName == predictor_name]
        fig.add_trace(go.Scatter(x=all_geo_df.Date,
                                 y=all_geo_df.PredictedDailyNewCases7DMA,
                                 name=predictor_name,
                                 visible=(ALL_GEO == DEFAULT_GEO))
                      )
        geoid_plot_names.append(ALL_GEO)

    # Add 1 trace per predictor, per geo id
    for predictor_name in predictor_names:
        for geoid_name in geoid_names:
            pred_geoid_df = ranking_df[(ranking_df.GeoID == geoid_name) &
                                       (ranking_df.PredictorName == predictor_name)]
            fig.add_trace(go.Scatter(x=pred_geoid_df.Date,
                                     y=pred_geoid_df.PredictedDailyNewCases7DMA,
                                     name=predictor_name,
                                     visible=(geoid_name == DEFAULT_GEO))
                          )
            geoid_plot_names.append(geoid_name)

    # For each geo
    # Add 1 trace for the true number of cases
    for geoid_name in geoid_names:
        geo_actual_df = actual_df[(actual_df.GeoID == geoid_name) &
                                  (actual_df.Date >= start_date)]
        fig.add_trace(go.Scatter(x=geo_actual_df.Date,
                                 y=geo_actual_df.ActualDailyNewCases7DMA,
                                 name="Ground Truth",
                                 visible=(geoid_name == DEFAULT_GEO),
                                 line=dict(color='orange', width=4, dash='dash'))
                      )
        geoid_plot_names.append(geoid_name)

    # Add 1 trace for the overall ground truth
    overall_actual_df = actual_df[actual_df.Date >= start_date].groupby(["Date"])[["GeoID", "ActualDailyNewCases7DMA"]].sum().     sort_values(by=["Date"]).reset_index()
    fig.add_trace(go.Scatter(x=overall_actual_df.Date,
                             y=overall_actual_df.ActualDailyNewCases7DMA,
                             name="Ground Truth",
                             visible= (ALL_GEO == DEFAULT_GEO),
                             line=dict(color='orange', width=4, dash='dash'))
                  )
    geoid_plot_names.append(selected_geo_id)

    # Format x axis
    fig.update_xaxes(
        dtick="D1",  # Means 1 day
        tickformat="%d\n%b")

    # Filter
    buttons=[]
    for geoid_name in ([ALL_GEO] + geoid_names):
        buttons.append(dict(method='update',
                            label=geoid_name,
                            args = [{'visible': [geoid_name==r for r in geoid_plot_names]},
                                    {'title': f"{geoid_name} Daily New Cases 7-day Average "}]))
    fig.update_layout(showlegend=True,
                      updatemenus=[{"buttons": buttons,
                                    "direction": "down",
                                    "active": ([ALL_GEO] + geoid_names).index(DEFAULT_GEO),
                                    "showactive": True,
                                    "x": 0.1,
                                    "y": 1.15}])

    fig.show()


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

    # Keep track of trace visibility by geo name
    ranking_geoid_plot_names = []

    all_df = ranking_df.groupby(["PredictorName", "Date"])[["GeoID", "PredictorName", "CumulDiff7DMA"]].sum().     sort_values(by=["PredictorName", "Date"]).reset_index()

    # Add 1 trace per predictor, for all geos
    for predictor_name in predictor_names:
        ranking_geoid_df = all_df[all_df.PredictorName == predictor_name]
        ranking_fig.add_trace(go.Scatter(x=ranking_geoid_df.Date,
                                         y=ranking_geoid_df.CumulDiff7DMA,
                                         name=predictor_name,
                                         visible=(ALL_GEO == DEFAULT_GEO))
                              )
        ranking_geoid_plot_names.append(ALL_GEO)


    # Add 1 trace per predictor, per country
    for predictor_name in predictor_names:
        for geoid_name in geoid_names:
            ranking_geoid_df = ranking_df[(ranking_df.GeoID == geoid_name) &
                                          (ranking_df.PredictorName == predictor_name)]
            ranking_fig.add_trace(go.Scatter(x=ranking_geoid_df.Date,
                                             y=ranking_geoid_df.CumulDiff7DMA,
                                             name=predictor_name,
                                             visible= (geoid_name == DEFAULT_GEO))
                                  )
            ranking_geoid_plot_names.append(geoid_name)

    # Format x axis
    ranking_fig.update_xaxes(
        dtick="D1",  # Means 1 day
        tickformat="%d\n%b")

    # Filter
    buttons=[]
    for geoid_name in ([ALL_GEO] + geoid_names):
        buttons.append(dict(method='update',
                            label=geoid_name,
                            args = [{'visible': [geoid_name==r for r in ranking_geoid_plot_names]},
                                    {'title': f'{geoid_name} submission rankings'}]))
    ranking_fig.update_layout(showlegend=True,
                              updatemenus=[{"buttons": buttons,
                                            "direction": "down",
                                            "active": ([ALL_GEO] + geoid_names).index(DEFAULT_GEO),
                                            "showactive": True,
                                            "x": 0.1,
                                            "y": 1.15}])

    return ranking_fig


def main():
    global app
    app.title = 'Cognizant COVID X-Prize'
    app.css.config.serve_locally = False
    # Don't be afraid of the 3rd party URLs: chriddyp is the author of Dash!
    # These two allow us to dim the screen while loading.
    # See discussion with Dash devs here: https://community.plotly.com/t/dash-loading-states/5687
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'})

    app.layout = html.Div(
        className="container scalable",
        children=[
            dcc.Store(id='memory'),
            html.Div(
                className="app_main_content",
                children=[
                    html.Div(
                        id="dropdown-select-outer",
                        className="row",
                        children=[
                            html.Div(
                                className="four columns",
                                children=[
                                    html.Div(
                                        children=[
                                            html.P(id='forecast_date', children='Forecast Date', hidden=True),
                                            dcc.Dropdown(
                                                id='selected_geo_id',
                                                options=[
                                                    {'label': 'overall', 'value': 'Overall'}
                                                ],
                                                value='Overall',
                                                style={"margin-bottom": "20px"}
                                            ),
                                        ], className="bigbox configbox"
                                    ),
                                ],
                            ),
                            html.Div(
                                [
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