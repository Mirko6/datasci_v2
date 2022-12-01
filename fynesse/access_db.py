from typing import List, Optional
from .config import *
from pymysql.connections import Connection

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""


PRICE_PAID_DATA_USEFUL_COLUMNS = [
  "price", "date_of_transfer", "postcode", "property_type"
]
POSTCODE_DATA_USEFUL_COLUMNS = [
  "postcode", "lattitude", "longitude", "status"
]
PRICE_POSTCODE_USEFUL_COLUMNS = list(set(PRICE_PAID_DATA_USEFUL_COLUMNS + POSTCODE_DATA_USEFUL_COLUMNS))


# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError


from pymysql import connect
import pandas as pd


def create_connection(user, password, host, database, port=3306):
      """ Create a database connection to the MariaDB database
          specified by the host url and database name.
      :param user: username
      :param password: password
      :param host: host url
      :param database: database
      :param port: port number
      :return: Connection object or None
      """
      conn = None
      try:
          conn = connect(user=user,
                          passwd=password,
                          host=host,
                          port=port,
                          local_infile=1,
                          db=database
                        )
      except Exception as e:
          print(f"Error connecting to the MariaDB Server: {e}")
      return conn


class DB:
  """
  Class designed to simplify execution of queries on the database.

  Methods
  -------
  __init__(conn):
    create a DB instance using a given connection

  custom_query(query):
    execute arbitrary SQL syntax query and return the result

  custom_select_query(query):
    execute arbitrary SELECT SQL syntax query and return 
    the result as a DataFrame

  select_top(table_name, n):
    Select top n elements from table table_name and return
    the result as a DataFrame

  select_priced_paid_data_joined_on_postcode(**kwargs):
    Join and Select priced paid data table with postcode data allowing for many filters
    Return the result as a DataFrame
  """
  def __init__(self, conn: Connection) -> None:
     self.conn = conn
     self.conn.ping()
     current_db = self.custom_query("SELECT database()")[0][0]
     tables = (table[0] for table in self.custom_query("SHOW TABLES"))
     print(f"you now have access to database: {current_db} with tables: {', '.join(tables)}")


  def custom_query(self, query: str):
    cur = self.conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    return result


  def custom_select_query(self, query: str) -> pd.DataFrame:
    cur = self.conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    column_names = [i[0] for i in cur.description]
    return pd.DataFrame.from_records(rows, columns=column_names)


  def select_top(self, table_name: str,  n: int = 5) -> pd.DataFrame:
    query = f'SELECT * FROM {table_name} LIMIT {n}'
    return self.custom_select_query(query)


  def select_priced_paid_data_joined_on_postcode(
      self,
      table_name_priced_paid_data: str = "pp_data",
      table_name_postcode_data: str = "postcode_data",
      date_from_incl: Optional[str] = None,
      date_to_excl: Optional[str] = None,
      town_city: Optional[str] = None,
      postcode: Optional[str] = None,
      longitude: Optional[str] = None,
      lattitude: Optional[str] = None,
      box_size: int = 0.01,
      only_live_postcodes : bool = False,
      property_type: Optional[str] = None,
      ppd_category_type: Optional[str] = 'A',
      columns_to_select: List[str] = [
        "price", "date_of_transfer", "property_type", "tenure_type", "new_build_flag", "country", "county", "town_city", "district", "longitude", "lattitude", "status", "ppd_category_type"
      ]
    ) -> pd.DataFrame:
      """
      Select Join and Select priced paid data table with postcode data allowing for many filters

      Args:
          Specifying tables args:
              table_name_priced_paid_data: Defaults to "pp_data".
              table_name_postcode_data: Defaults to "postcode_data".

          Positional filters arg:
              longitude: longitude coordinate of the center of a box.
              lattitude: lattitude coordinate of the center of a box.
              box_size: create a filter for transactions using a square of size box_size with centre in (longitude, lattitude). 
                        Defaults to 0.01. Which corresponds to ~1.1km.

          Additional filters args:
              date_from_incl: lower bound filter on date. Expected format is yyyy/mm/dd.
              date_from_excl: upper bound filter on date. Expected format is yyyy/mm/dd.
              town_city: town or city to filter for.
              postcode: postcode to filter for.
              only_live_postcodes: there are two values for postcode status - live and terminated.
                                  True will filter away terminated status. Default False.
              ppd_category_type: Defaults to 'A'.

          Other args:
              columns_to_select: Columns to select in addition to postcode. 
                                Defaults to [price, date_of_transfer, property_type, tenure_type, new_build_flag, country, county,
                                            town_city, district, longitude, lattitude, postcode_status, ppd_category_type]

      Returns:
          DataFrame with corresponding data
      """      
      filters = []
      if date_from_incl is not None:
        filters.append(f"'{date_from_incl}' <= date_of_transfer")
      if date_to_excl is not None:
        filters.append(f"date_of_transfer < '{date_to_excl}'")
      if town_city is not None:
        filters.append(f"town_city = '{town_city.upper()}'")
      if postcode is not None:
        filters.append(f"{table_name_priced_paid_data}.postcode = '{postcode}'")
      if longitude is not None:
        longitude = float(longitude)
        filters.append(f"{longitude - box_size/2} < longitude AND longitude < {longitude + box_size/2}")
      if lattitude is not None:
        lattitude = float(lattitude)
        filters.append(f"{lattitude - box_size/2} < lattitude AND lattitude < {lattitude + box_size/2}")
      if only_live_postcodes:
        filters.append("status = 'live'")
      if property_type is not None:
        filters.append(f"property_type = '{property_type}'")
      if ppd_category_type is not None:
        filters.append(f"ppd_category_type = '{ppd_category_type}'")

      columns_to_select.append(f"{table_name_priced_paid_data}.postcode")
      query = f"""
        SELECT {', '.join(columns_to_select)}
          FROM {table_name_priced_paid_data}
          JOIN {table_name_postcode_data} 
            ON {table_name_priced_paid_data}.postcode = {table_name_postcode_data}.postcode
         WHERE {' AND '.join(filters)}
      """
      print(f"the following query gets executed: {query}")
      return self.custom_select_query(query)
