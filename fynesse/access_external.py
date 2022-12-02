#this file provides functions for fetching and uploading the priced paid data and the postcode data to the database

from typing import Optional
from .config import *
from pymysql.connections import Connection
import pandas as pd
from os import remove
from requests import get
from zipfile import ZipFile
from io import BytesIO


FIRST_YEAR_PP_DATA = 1995
CURRENT_YEAR = 2022

# general functions

def upload_csv_to_aws(conn: Connection, table_name: str, file_name: str):
  cur = conn.cursor()
  query = f"""
    LOAD DATA LOCAL INFILE '{file_name}' INTO TABLE {table_name}
    FIELDS TERMINATED BY ',' 
    LINES STARTING BY '' TERMINATED BY '\n'
  """
  cur.execute(query)
  rows_affected=cur.rowcount
  return rows_affected


def create_index_using_hash_if_not_exists(conn: Connection, table_name: str, index_col: str, index_name: Optional[str] = None):
  if index_name is None:
    index_name = table_name + "." + index_col
  cur = conn.cursor()
  query = f"""
    CREATE INDEX IF NOT EXISTS `{index_name}` 
    USING HASH ON `{table_name}` ({index_col})
  """
  cur.execute(query)


# priced paid data functions

def get_data(year: int, part: int):
  url = f"http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}-part{part}.csv"
  df = pd.read_csv(url, header=None)
  return df


def create_table_and_upload_everything_for_priced_paid_data(
  conn: Connection,
  table_name: str = 'pp_data',
  sample_table_name: str = 'pp_data_sample',
  sample_portion = 0.01,
  first_year: int = FIRST_YEAR_PP_DATA,
  current_year: int = CURRENT_YEAR,
):
  drop_and_create_table_for_priced_paid_data(conn, table_name)
  drop_and_create_table_for_priced_paid_data(conn, sample_table_name)

  upload_datasets(conn, table_name, sample_table_name, first_year, current_year, sample_portion)
  upload_current_year_dataset(conn, table_name, sample_table_name, sample_portion, current_year)


def upload_datasets(
    conn: Connection, 
    table_name: str, 
    sample_table_name: str, 
    year_from: int, 
    year_to_excl: int, 
    sample_portion = 0.01
  ):
    num_fetched_rows = []
    num_uploaded_rows = []
    for year in range(year_from, year_to_excl):
      for part in [1, 2]:
        print(f"fetching data from year: {year}, part: {part}")
        df = get_data(year, part)
        num_fetched_rows.append(len(df))
        filename = f"price_paid_data_year_{year}_part_{part}.csv"
        df.to_csv(filename, header=False, index=False)
        df.sample(frac=sample_portion).to_csv("sample_" + filename, header=False, index=False)
        print(f"uploading the data")
        num_uploaded_rows.append(
            upload_csv_to_aws(conn, table_name, filename)
        )
        upload_csv_to_aws(conn, sample_table_name, "sample_" + filename)
        remove(filename)
        remove("sample_" + filename)
    print(f"num_fetched_rows was {sum(num_fetched_rows)}, num_uploaded_rows was {sum(num_uploaded_rows)}")


def upload_current_year_dataset(conn, table_name="pp_data", sample_table_name="pp_data_sample", sample_portion=0.01, current_year=CURRENT_YEAR):
  print(f"fetching data from year: {current_year}")
  url = f"http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{current_year}.csv"
  df = pd.read_csv(url, header=None)
  num_fetched = len(df)
  dataset_path = f"price_paid_data_year_{CURRENT_YEAR}.csv"
  sample_dataset_path = f"sample_price_paid_data_year_{CURRENT_YEAR}.csv"
  df.to_csv(dataset_path, header=False, index=False)
  df.sample(frac=sample_portion).to_csv(sample_dataset_path, header=False, index=False)
  print("uploading data")
  num_uploaded = upload_csv_to_aws(conn, table_name, dataset_path)
  upload_csv_to_aws(conn, sample_table_name, sample_dataset_path)
  remove(dataset_path)
  remove(sample_dataset_path)
  print(f"num_fetched_rows was {num_fetched}, num_uploaded_rows was {num_uploaded}")


def drop_and_create_table_for_priced_paid_data(conn: Connection, table_name: str = 'pp_data'):
  cur = conn.cursor()
  query1 = f"DROP TABLE IF EXISTS `{table_name}`"
  query2 = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
      `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
      `price` int(10) unsigned NOT NULL,
      `date_of_transfer` date NOT NULL,
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
      `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `street` tinytext COLLATE utf8_bin NOT NULL,
      `locality` tinytext COLLATE utf8_bin NOT NULL,
      `town_city` tinytext COLLATE utf8_bin NOT NULL,
      `district` tinytext COLLATE utf8_bin NOT NULL,
      `county` tinytext COLLATE utf8_bin NOT NULL,
      `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
      `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
      PRIMARY KEY (`db_id`)
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
  """

  cur.execute(query1)
  cur.execute(query2)


# postcode data functions


def drop_and_create_table_for_postcode_data(conn: Connection, table_name: str = "postcode_data"):
  cur = conn.cursor()
  query1 = f"DROP TABLE IF EXISTS `{table_name}`"
  query2 = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
        `status` enum('live','terminated') NOT NULL,
        `usertype` enum('small', 'large') NOT NULL,
        `easting` int unsigned,
        `northing` int unsigned,
        `positional_quality_indicator` int NOT NULL,
        `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
        `lattitude` decimal(11,8) NOT NULL,
        `longitude` decimal(10,8) NOT NULL,
        `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
        `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
        `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
        `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
        `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
        `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
        `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
        `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
        `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
        PRIMARY KEY (`db_id`)
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
  """

  cur.execute(query1)
  cur.execute(query2)


def fetch_and_upload_postcode_data(conn: Connection, table_name = 'postcode_data'):
  url = "https://www.getthedata.com/downloads/open_postcode_geo.csv.zip"
  r = get(url)
  z = ZipFile(BytesIO(r.content))
  z.extract("open_postcode_geo.csv")
  z.extract("licence.txt")
  with open("licence.txt") as f:
    print("licence of the postcode data:")
    print(f.read())
  remove("licence.txt")
  upload_csv_to_aws(conn, table_name, 'open_postcode_geo.csv')
  remove("open_postcode_geo.csv")


