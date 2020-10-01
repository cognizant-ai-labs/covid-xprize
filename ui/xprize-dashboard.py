import plotly.graph_objects as go

ALL_GEO = "Overall"
DEFAULT_GEO = ALL_GEO

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

all_df = ranking_df.groupby(["PredictorName", "Date"])[["GeoID", "PredictorName", "PredictedDailyNewCases7DMA"]].sum().     sort_values(by=["PredictorName", "Date"]).reset_index()
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
                             visible= (geoid_name == DEFAULT_GEO),
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
geoid_plot_names.append(geoid_name)

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


# ## Rankings: by cumulative 7DMA error

# In[ ]:


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

ranking_fig.show()