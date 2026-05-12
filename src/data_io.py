import pandas as pd
import sqlite3


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_to_sqlite(df: pd.DataFrame, table: str, db_path: str,
                   if_exists: str = 'replace') -> None:
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists=if_exists, index=False)
    conn.close()
    print(f"Saved {len(df):,} rows → {table} ({db_path})")


def load_from_sqlite(query: str, db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df