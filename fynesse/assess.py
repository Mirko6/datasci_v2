from typing import Optional, Tuple
from .config import *

from . import access_db
import pandas as pd
from datetime import date, timedelta


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
  df_counts = pd.DataFrame({"counts": counts})
  df_counts["percentage"] = df_counts["counts"].apply(lambda c: round(100*c/df_counts['counts'].sum(), 1))
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
