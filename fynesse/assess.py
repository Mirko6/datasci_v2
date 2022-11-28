from typing import Optional, Tuple
from .config import *

from . import access_db
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import osmnx as ox
import networkx
import geopandas as gpd

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


def plot_price_distribution(prices: pd.Series, bin_width: int = 100_000, cmap: str = 'plasma'):
  bin_low = math.floor(prices.min() / bin_width) * bin_width
  bin_high = math.ceil(prices.max() / bin_width) * bin_width

  bins = np.linspace(bin_low, bin_high, num = (bin_high - bin_low) // bin_width + 1)
  
  n, bins, patches = plt.hist(prices, bins=bins)

  #cmap scaling
  bin_centers = 0.5 * (bins[:-1] + bins[1:])
  col = bin_centers - min(bin_centers)
  col /= max(col)
  color_map = plt.cm.get_cmap(cmap)
  for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', color_map(c))

  plt.ylabel('Count')
  plt.xlabel('Price')
  plt.title('Counts per price_level')


def fetch_graph_from_df(df: pd.DataFrame) -> networkx.MultiDiGraph:
  return ox.graph_from_bbox(df['lattitude'].min(), df['lattitude'].max(), df['longitude'].min(),  df['longitude'].max())


#TODO: maybe return just an axis?
def plot_prices_on_map(df: pd.DataFrame, edges: gpd.GeoDataFrame):
  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.lattitude))
  #fetching geographic data
  if edges is None:
    graph = ox.graph_from_bbox(df['lattitude'].min(), df['lattitude'].max(), df['longitude'].min(),  df['longitude'].max())
    nodes, edges = ox.graph_to_gdfs(graph)

  #plotting
  fig, ax = plt.subplots(figsize=(12, 12))
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray", zorder=1)
  ax.set_xlim([df['longitude'].min(), df['longitude'].max()])
  ax.set_ylim([df['lattitude'].min(), df['lattitude'].max()])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")
  gdf.sort_values(by=['price']).plot("price", ax=ax, legend=True, cmap='plasma', alpha=0.7, zorder=2) #use sort_values, so the lighter colors are drawn later
  return fig, ax


def get_bbox_from_df(df) -> Tuple[float, float, float, float]:
  return df['lattitude'].min(), df['lattitude'].max(), df['longitude'].min(),  df['longitude'].max()