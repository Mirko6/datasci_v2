from .config import *
from pymysql.connections import Connection

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

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


class DB_access:
  def __init__(self, conn: Connection) -> None:
     self.conn = conn
     self.conn.ping()
     current_db = self.custom_query("SELECT database()")[0][0]
     tables = (table[0] for table in self.custom_query("SHOW TABLES"))
     print(f"you now have access to database {current_db} with tables {', '.join(tables)}")


  def _select_top(self, table_name: str,  n: int = 5):
      cur = self.conn.cursor()
      cur.execute(f'SELECT * FROM {table_name} LIMIT {n}')

      rows = cur.fetchall()
      column_names = [i[0] for i in cur.description]
      return rows, column_names


  def select_top(self, table_name: str,  n: int = 5) -> pd.DataFrame:
    rows, column_names = self._select_top(table_name, n)
    df = pd.DataFrame.from_records(rows, columns=column_names)
    return df


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
