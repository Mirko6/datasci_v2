from typing import Optional, Tuple
from .config import *

from . import access_db
import pandas as pd
from datetime import date, timedelta

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
