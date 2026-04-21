import sqlite3
import pandas as pd
conn = sqlite3.connect('sow.db')
sow_id = 12  # Replace with the actual SOW ID you want to query

#df = pd.read_sql_query("SELECT * FROM sow WHERE id=?", conn, params=(sow_id,))
df = pd.read_sql_query("SELECT * FROM sow ", conn)
print(df.head(10))

# if sow_id:
#         cursor.execute("SELECT * FROM sow WHERE id=?", (sow_id,))
#     else:
#         cursor.execute("SELECT * FROM sow")

