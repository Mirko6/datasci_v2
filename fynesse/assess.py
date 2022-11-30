from typing import Optional, Tuple
from .config import *

from . import access_db
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import osmnx as ox
import networkx
import geopandas as gpd
from shapely.geometry import Point
from datetime import date, datetime, timedelta

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access_db.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def get_training_data(
  DB,
  lattitude: float,
  longitude: float,
  date_prediction: date,
  property_type: Optional[str] = None,
  days_to_consider: int = 365,
  box_size: float = 0.02
  ):
  date_max_from_data = DB.custom_select_query("SELECT max(date_of_transfer) as max_date FROM pp_data")['max_date'][0]
  date_to = min(date_max_from_data + timedelta(days=1), date_prediction)
  date_from = date_to - timedelta(days = days_to_consider)
  df_all = DB.select_priced_paid_data_joined_on_postcode(
    date_from_incl=date_from.strftime("%Y/%m/%d"), 
    date_to_excl=date_to.strftime("%Y/%m/%d"),
    table_name_priced_paid_data="pp_data", 
    table_name_postcode_data="postcode_data",
    longitude = longitude,
    lattitude = lattitude,
    box_size = box_size,
    only_live_postcodes=True,
    property_type = property_type,
  )
  return df_all

def filter_price_outliers(df: pd.DataFrame, fraction_to_remove = 0.1):
  print(f"Number of all houses: {len(df)}, price range: {df['price'].min()} - {df['price'].max()}")
  one_sided_outliers = fraction_to_remove / 2
  low_price = df['price'].quantile(one_sided_outliers, interpolation='nearest')
  high_price = df['price'].quantile(1 - one_sided_outliers, interpolation='nearest')
  df_filtered = df[
      (low_price <= df['price']) &
      (df['price'] <= high_price)
  ]
  print(f"Number of houses without outliers: {len(df_filtered)}, price range: {df_filtered['price'].min()} - {df_filtered['price'].max()}")
  return df_filtered


def fetch_graph_from_df(df: pd.DataFrame) -> networkx.MultiDiGraph:
  return ox.graph_from_bbox(df['lattitude'].min(), df['lattitude'].max(), df['longitude'].min(),  df['longitude'].max())


def get_bbox_from_df(df, delta_coordinates = 0.001) -> Tuple[float, float, float, float]:
  return (
    float(df['lattitude'].min()) - delta_coordinates, #north
    float(df['lattitude'].max()) + delta_coordinates, #south
    float(df['longitude'].min()) - delta_coordinates, #east
    float(df['longitude'].max()) + delta_coordinates  #west
  )


def num_objects_within_d(object_geometries, d_within: float, point: Point):
  return sum(object_geometries.apply(lambda geometry: geometry.distance(point) < d_within ))


def _round_to_base(x, base):
  return base * round(x/base)


def plot_price_distribution(
  prices: pd.Series,
  ax: Axes,
  bin_width: Optional[int] = None,
  cmap: str = 'plasma'
  ):
  APPROXIMATE_NUMBER_OF_BINS = 7
  BIN_WIDTH_IS_DIVISIBLE_BY = 25_000
  if bin_width is None:
    bin_width = _round_to_base(
      (prices.max() - prices.min()) / APPROXIMATE_NUMBER_OF_BINS, 
      BIN_WIDTH_IS_DIVISIBLE_BY
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
  ) -> gpd.DataFrame:
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
  glm_result, 
  num_division_per_dimension: int = 150,
  include_legend: bool = True,
  ):
  x = np.linspace(float(df_train['longitude'].min()), float(df_train['longitude'].max()), num_division_per_dimension)
  y = np.linspace(float(df_train['lattitude'].min()), float(df_train['lattitude'].max()), num_division_per_dimension)
  xx, yy = np.meshgrid(x, y)
  df_box_predictions = pd.DataFrame({'longitude': xx.flatten(), 'lattitude': yy.flatten()})
  prediction_cols = (x for x in df_values_for_prediction.columns if x not in ['lattitude', 'longitude'])
  for col_name in prediction_cols:
    df_box_predictions[col_name] = df_values_for_prediction[col_name][0]
  
  df_box_predictions['price'] = df_box_predictions.apply(lambda row: glm_result.predict(pd.DataFrame(row).transpose()).iloc[0], axis=1)
  gdf = gpd.GeoDataFrame(df_box_predictions, geometry=gpd.points_from_xy(df_box_predictions.longitude, df_box_predictions.lattitude))
  gdf.plot('price', ax=ax_price_colormap, vmin=df_train['price'].min(), vmax=df_train['price'].max(), cmap='plasma', zorder=1, legend=include_legend)
  return gdf