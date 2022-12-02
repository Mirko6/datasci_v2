from typing import Dict, Optional, Tuple
from .config import *

from . import access_db
import pandas as pd
import geopandas as gpd
from datetime import date, timedelta
from matplotlib.axes import Axes
import osmnx as ox


def get_training_data(
  DB: access_db.DB,
  lattitude: float,
  longitude: float,
  date_prediction: date,
  date_from: Optional[date] = None,
  date_to: Optional[date] = None,
  property_type: Optional[str] = None,
  days_to_consider: int = 365, #used when date_from is None
  box_size: float = 0.02
  ):
  if date_to is None:
    date_max_from_data = DB.custom_select_query("SELECT max(date_of_transfer) as max_date FROM pp_data")['max_date'][0]
    date_to = min(date_max_from_data + timedelta(days=1), date_prediction)
  if date_from is None:
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
  print(f"Number of houses before filtering: {len(df)}, price range: {df['price'].min()} - {df['price'].max()}")
  one_sided_outliers = fraction_to_remove / 2
  low_price = df['price'].quantile(one_sided_outliers, interpolation='nearest')
  high_price = df['price'].quantile(1 - one_sided_outliers, interpolation='nearest')
  df_filtered = df[
      (low_price <= df['price']) &
      (df['price'] <= high_price)
  ]
  print(f"Number of houses without price outliers: {len(df_filtered)}, price range: {df_filtered['price'].min()} - {df_filtered['price'].max()}")
  return df_filtered


def occurence_of_values_in_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
  """Given a dataframe and a name of one of its columns, outputs an overview dataframe
  showing the counts and percentages of value occurences of the given column.

  Returns:
      pd.DataFrame: Indexed by the set of values in the column of interest.
                    Containing columns counts and percentage.
  """  
  counts = df[column_name].groupby(by=df[column_name]).count().sort_values(ascending=False)
  df_counts = pd.DataFrame({"count": counts})
  df_counts["percentage"] = df_counts["count"].apply(lambda c: round(100*c/df_counts['count'].sum(), 1))
  return df_counts


def get_bbox(longitude: str, lattitude: str, box_size: float) -> Tuple[float, float, float, float]:
  """given coordinates of a point and a box_size, returns bounds of the
  square of size box_size centered at that point with

  Returns:
      Tuple[float, float, float, float]: north, south, east, west coordinate bounds respectively
  """  
  longitude, lattitude = float(longitude), float(lattitude)
  north = lattitude + box_size / 2
  south = lattitude - box_size / 2
  east = longitude + box_size / 2
  west = longitude - box_size / 2
  return north, south, east, west


def plot_edges(bbox: Tuple[float, float, float, float], ax: Axes) -> None:
  """Fetches and plots edges within bbox on the axis ax"""  
  graph = ox.graph_from_bbox(*bbox)
  _nodes, edges = ox.graph_to_gdfs(graph)
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")


def plot_pois(
  bbox: Tuple[float, float, float, float],
  ax: Axes,
  tags: Dict[str, str],
  color: Optional[str] = None
  ) -> None:
  """Fetches and plots points of interests defined by the tags within bbox on the axis ax"""  
  pois = ox.geometries_from_bbox(*bbox, tags)
  if len(pois) > 0:
    pois.plot(ax=ax, color=color, alpha=0.7, markersize=10)
  else:
    print(f"There are no points of interest for tags {tags} in the given bbox: {bbox}")


def tags_in_pois_occurences(pois: gpd.GeoDataFrame, tags: Dict[str, str]) -> None:
  """Given points of interests and tags, prints out the numerical information about the occurences of tags"""  
  for key in tags.keys():
    print()
    if key in pois.columns:
      print(f"For tag: {key}, we have the following values in pois:")
      print(occurence_of_values_in_column(pois, key))
    else:
      print(f"For tag: {key}, we have no pois")
