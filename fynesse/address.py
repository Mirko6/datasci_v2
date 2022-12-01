# This file contains code for suporting addressing questions in the data

from typing import Optional, Tuple
from .config import *

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import osmnx as ox
import networkx
import geopandas as gpd
from shapely.geometry import Point
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import statsmodels as sm
from datetime import date
#pd.options.mode.chained_assignment = None

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

def predict_price(
    df: pd.DataFrame,
    lattitude: str,
    longitude: str,
    predict_date: date,
    prediction_features,
  ) -> Tuple[Optional[gpd.GeoDataFrame], float, GLMResultsWrapper, pd.DataFrame]:
    pois = None
    df_design_GLM = df[['longitude', 'lattitude']].astype({'longitude': float, 'lattitude': float})
    df_values_for_prediction = pd.DataFrame(data={'longitude': [float(longitude)], 'lattitude': [float(lattitude)]})
    if prediction_features.get("constant"):
      df_design_GLM['constant'] = 1
      df_values_for_prediction['constant'] = 1
    if prediction_features.get("new_build_flag") is not None:
      df_design_GLM['new_build_flag'] = (df['new_build_flag'] == 'N').astype(int)
      df_values_for_prediction['new_build_flag'] = int(prediction_features.get("new_build_flag"))
    if prediction_features.get("days_since"):
      if prediction_features.get("days_since") == "first_day":
        df.loc[:, 'ordinal_date_value'] = df['date_of_transfer'].apply(lambda d: d.toordinal())
        df.loc[:, 'days_since_first_day'] = df['ordinal_date_value'] - df['ordinal_date_value'].min()
        df_design_GLM['days_since_first_day'] = df['days_since_first_day']
        df_values_for_prediction['days_since_first_day'] = (predict_date - df['date_of_transfer'].min()).days
      else:
        year, month, day = (int(i) for i in prediction_features.get("days_since").split("/"))
        date_since = date(year, month, day)
        df.loc[:, 'ordinal_date_value'] = df['date_of_transfer'].apply(lambda d: d.toordinal())
        df.loc[:, 'days_since_date'] = df['ordinal_date_value'] - date_since.toordinal()
        df_design_GLM['days_since_date'] = df['days_since_date']
        df_values_for_prediction['days_since_date'] = (predict_date - date_since).days    
    if prediction_features.get("num_objects"):
      d_within = prediction_features["num_objects"]["d_within"]
      tags = prediction_features["num_objects"]["tags"]
      pois = ox.geometries_from_bbox(*get_bbox_from_df(df), tags)
      df.loc[:, 'num_objects'] = df.apply(lambda row: num_objects_within_d(pois['geometry'], d_within, Point(row['longitude'], row['lattitude'])), axis = 1)
      df_design_GLM['num_objects'] = df['num_objects']
      df_values_for_prediction['num_objects'] = num_objects_within_d(pois['geometry'], d_within, Point(float(longitude), float(lattitude)))
    if not prediction_features.get("longitude"):
      df_design_GLM.drop(columns='longitude', inplace=True)
      df_values_for_prediction.drop(columns='longitude', inplace=True)
    if not prediction_features.get("lattitude"):
      df_design_GLM.drop(columns = 'lattitude', inplace=True)
      df_values_for_prediction.drop(columns='lattitude', inplace=True)

    print(f"GLM uses the following features: {', '.join(df_design_GLM.columns)}")

    glm = sm.GLM(df['price'], df_design_GLM)
    glm_result = glm.fit()

    prediction = glm_result.predict(df_values_for_prediction)[0]
    std = math.sqrt(glm_result.scale)
    print(f"Price prediction: {round(prediction)}£, std: {round(std)}, 95% confidence interval: {round(prediction - 2*std)}£ - {round(prediction + 2*std)}£")

    return pois, prediction, glm_result, df_values_for_prediction


def fetch_graph_from_df(df: pd.DataFrame) -> networkx.MultiDiGraph:
  return ox.graph_from_bbox(df['lattitude'].min(), df['lattitude'].max(), df['longitude'].min(),  df['longitude'].max())


def get_bbox_from_df(df, delta_coordinates = 0.001) -> Tuple[float, float, float, float]:
  return (
    float(df['lattitude'].min()) - delta_coordinates, #north
    float(df['lattitude'].max()) + delta_coordinates, #south
    float(df['longitude'].min()) - delta_coordinates, #east
    float(df['longitude'].max()) + delta_coordinates  #west
  )


#brutforce implemntation - might make it faster if wanted, but since there is usually small number of geometries it is okay
def num_objects_within_d(object_geometries, d_within: float, point: Point):
  return sum(object_geometries.apply(lambda geometry: geometry.distance(point) < d_within ))


def _round_to_base(x, base):
  return base * round(x/base)


def plot_price_distribution(
  prices: pd.Series,
  ax: Axes,
  bin_width: Optional[int] = None,
  cmap: str = 'plasma',
  approximate_number_of_bins: float = 9, #used only when bin_width is None
  bin_width_is_divisible_by: float = 20_000, #used only when bin_width is None
  ):
  if bin_width is None:
    bin_width = _round_to_base(
      (prices.max() - prices.min()) / approximate_number_of_bins, 
      bin_width_is_divisible_by
    )

  bin_low = math.floor(prices.min() / bin_width) * bin_width
  bin_high = math.ceil(prices.max() / bin_width) * bin_width

  bins = np.linspace(bin_low, bin_high, num = (bin_high - bin_low) // bin_width + 1)
  
  _n, bins, patches = ax.hist(prices, bins=bins)

  #cmap scaling
  bin_centers = 0.5 * (bins[:-1] + bins[1:])
  col = bin_centers - min(bin_centers)
  col /= max(col)
  color_map = plt.cm.get_cmap(cmap)
  for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', color_map(c))

  ax.set_ylabel('Count')
  ax.set_xlabel('Price')
  ax.set_title('Counts per price_level')


def plot_prices_on_map(
    df: pd.DataFrame,
    ax: Axes,
    edges: Optional[gpd.GeoDataFrame] = None,
    min_markersize: float = 3,
    markersize_count_scaling_constant: float = 200,
    include_legend: bool = True,
    colormap = 'plasma',
  ) -> gpd.GeoDataFrame:
  df_by_postcode = (df.groupby(by='postcode')
            .aggregate({'price': 'mean', 'date_of_transfer': 'count', 'lattitude': 'median', 'longitude': 'median'})
            .rename(columns={'date_of_transfer':'count'}))
  gdf = gpd.GeoDataFrame(df_by_postcode, geometry=gpd.points_from_xy(df_by_postcode.longitude, df_by_postcode.lattitude))

  #fetching geographic data
  if edges is None:
    graph = ox.graph_from_bbox(*get_bbox_from_df(gdf))
    _nodes, edges = ox.graph_to_gdfs(graph)

  #plotting
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray", zorder=1)
  north, south, east, west = get_bbox_from_df(df)
  ax.set_xlim([east, west])
  ax.set_ylim([north, south])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")
  gdf = gdf.sort_values(by=['price'])
  gdf.plot( #use sort_values, so the lighter colors are drawn later
      "price",
      ax=ax,
      markersize= min_markersize + markersize_count_scaling_constant * gdf['count']/gdf['count'].max(), #markersize based on the count of values at given location
      legend=include_legend,
      cmap=colormap,
      vmin=df['price'].min(),
      vmax=df['price'].max(),
      alpha=0.7,
      zorder=2
    )
  return gdf


def plot_price_prediction_color_map(   
  df_train: pd.DataFrame, 
  ax_price_colormap: Axes,
  df_values_for_prediction: pd.DataFrame,
  glm_result: GLMResultsWrapper,
  num_division_per_dimension: int = 150,
  include_legend: bool = True,
  color_scale_based_on_df_train: bool = True,
  cmap: str = 'plasma',
  ) -> gpd.GeoDataFrame:
  x = np.linspace(float(df_train['longitude'].min()), float(df_train['longitude'].max()), num_division_per_dimension)
  y = np.linspace(float(df_train['lattitude'].min()), float(df_train['lattitude'].max()), num_division_per_dimension)
  xx, yy = np.meshgrid(x, y)
  df_box_predictions = pd.DataFrame({'longitude': xx.flatten(), 'lattitude': yy.flatten()})
  for col_name in df_values_for_prediction.columns:
    if col_name not in ['lattitude', 'longitude']:
      # we want to keep constant all other features other than lattitude and longitude
      df_box_predictions[col_name] = df_values_for_prediction[col_name][0]

  # predict prices from the glm_result for each row
  df_box_predictions['price'] = df_box_predictions.apply(
    lambda row: glm_result.predict(
        pd.DataFrame(row).transpose()[df_values_for_prediction.columns]
      ).iloc[0],
    axis=1
  )
  gdf = gpd.GeoDataFrame(df_box_predictions, geometry=gpd.points_from_xy(df_box_predictions.longitude, df_box_predictions.lattitude))
  
  vmin, vmax = (df_train['price'].min(), df_train['price'].max()) if color_scale_based_on_df_train else (None, None)
  gdf.plot('price', ax=ax_price_colormap, vmin=vmin, vmax=vmax, cmap=cmap, zorder=1, legend=include_legend)
  return gdf