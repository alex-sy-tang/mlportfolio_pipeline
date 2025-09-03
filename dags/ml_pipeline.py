import requests
from datetime import datetime, timedelta
import pandas as pd 
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import SQLExecuteQueryOperator
from airflow.hooks.base import BaseHook

# set up default arguments for dag
load_dotenv()



# Alpha Vantage API: 

alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
etf_symbol = "QQQ"

# Financial Modeling API:
base_url = "https://financialmodelingprep.com/api"
data_type = "historical-price-full"
ticker = "AAPL"

fmp_api_key = os.getenv("FMP_API_KEY")
fmp_url = f"{base_url}/v3/{data_type}/{ticker}?apikey={fmp_api_key}"


#---------------------------------------------
# 1. Get Data from the alpha-vantage api
def get_qqq_holdings(api_key: str, symbol: str) -> pd.DataFrame:

    url = f'https://www.alphavantage.co/query?function=ETF_PROFILE&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200: 
        json_data = response.json()
        holdings_data = json_data["holdings"]
        df = pd.DataFrame(holdings_data)
        # df.to_csv('/opt/airflow/data/raw/qqq_holdings_raw.csv', index = False)
    else: 
        print("API request to Alpha Vantage failed.")

    holdings_list = df['symbol'].to_list()
    return holdings_list


def get_historical_price(url: str) -> pd.DataFrame: 
    response = requests.get(url)
    data = response.json()
    symbol, historical_price = data['symbol'], data['historical']
    
    df_price = pd.DataFrame(historical_price)
    s_symbol = pd.Series(symbol, range(len(historical_price))).rename("Ticker")
    df_symbol = pd.DataFrame(s_symbol)
    
    df_sorted = pd.concat([df_symbol, df_price], axis = 1).sort_values(by = "date")
    df = df_sorted.reset_index(drop = True)
    df = df.drop(columns=['label'])

    # df.to_csv('/opt/airflow/logs/historical_price.csv', 
    #     index = False)
    
    print("Got historical price successfully")

    return df


def get_historical_prices(ti, api_key): 

    base_url = "https://financialmodelingprep.com/api"
    data_type = "historical-price-full"
    
    df_agg = pd.DataFrame()
    
    tickers = ti.xcom_pull(task_ids = 'get_qqq_holdings')

    for i in range(len(tickers[:2])):
        url = f"{base_url}/v3/{data_type}/{tickers[i]}?apikey={api_key}"
        df_temp = get_historical_price(url)
        df_agg = pd.concat([df_agg, df_temp], axis = 0)

    df_agg.to_csv('/opt/airflow/logs/historical_prices.csv', index = False)

    print("Got aggregated historical price successfully!")

#---------------------------------------------
#2.Store data into the postgresql server

historical_price_file = '/opt/airflow/logs/historical_prices.csv' 
def load_price_data(file):
    with open(file , 'r') as reader: 
        lines = reader.readlines()
        data = [line.split(",") for line in lines]
        return data[1:]

def save_to_database(file):
    data = load_price_data(file)
    hook = BaseHook.get_connection('postgresql')
    conn = psycopg2.connect(
        host = hook.host, 
        user = hook.login, 
        password = hook.password, 
        database = hook.schema, 
        port = hook.port
    )

    try: 
        cursor = conn.cursor()
        # print('Connected to the PostgreSQL database')

        query = """
            INSERT INTO historical_prices (Ticker, Date, open, high, low, close, adjClose, volume, unadjustedVolume, change, changePercent, vwap, changeOverTime)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.executemany(query, data)
        conn.commit()

    except psycopg2.Error as e:
        print(f"An Error has occured: {e}")

    finally: 
        if conn: 
            print(cursor.rowcount, "record inserted. ")
            cursor.close()
            conn.close()

    

#---------------------------------------------
#default arguments: 

default_args = {
    'owner' : 'yao', 
    'retries' : 5, 
    'retry_delay' : timedelta(minutes = 5)
}


#---------------------------------------------
# DAG RUNS
with DAG(
    default_args = default_args, 
    dag_id = 'ml_pipeline_v01', 
    description = 'A DAG for ml pipeline', 
    start_date = datetime(2025, 8, 23), 
    schedule = '@daily'
) as dag: 

    get_holdings = PythonOperator(
        task_id = 'get_qqq_holdings', 
        python_callable = get_qqq_holdings, 
        op_kwargs = {'api_key': alpha_vantage_api_key, 'symbol': etf_symbol}
    )

    # get_price = PythonOperator(
    #     task_id = 'get_historical_price', 
    #     python_callable = get_historical_price, 
    #     op_kwargs = {'url': fmp_url}
    # )

    create_tables = SQLExecuteQueryOperator(
        task_id = 'create_tables', 
        conn_id = 'postgresql', 
        sql="""
            create table if not exists historical_prices (
                Ticker varchar(30), 
                Date date, 
                open float, 
                high float,
                low float,
                close float, 
                adjClose float, 
                volume int, 
                unadjustedVolume int, 
                change float, 
                changePercent float, 
                vwap float, 
                changeOverTime float, 
                primary key (Ticker, Date)
            )
        """
    )

    save_historical_price = PythonOperator(
        task_id = 'save_historical_price', 
        python_callable = save_to_database, 
        op_kwargs = {'file': historical_price_file}
    )

    get_prices = PythonOperator(
        task_id = 'get_historical_prices', 
        python_callable = get_historical_prices, 
        op_kwargs = {'api_key':fmp_api_key}
    )

    get_holdings >> get_prices >> create_tables >> save_historical_price