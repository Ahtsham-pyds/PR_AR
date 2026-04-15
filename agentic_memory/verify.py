import sqlite3
import pandas as pd
conn = sqlite3.connect('sow.db')
df = pd.read_sql_query("SELECT * FROM sow", conn)
df.head()

