from ligo.gracedb.rest import GraceDb
import pandas as pd
from astropy.table import Table
from astropy.time import Time
import warnings
from tqdm import tqdm

gracedb = GraceDb()
query_result = gracedb.superevents(query='created: 2024-01-01 .. 2024-01-31')#, columns=['superevent_id', 't_start', 't_0', 't_end', 'far'])

df = pd.DataFrame([dict(s) for s in tqdm(query_result)])
df['t_0'] = df['t_0'].apply(lambda gps: Time(gps, format='gps', scale='utc').to_value('iso', subfmt='date_hms'))
df['t_start'] = df['t_start'].apply(lambda gps: Time(gps, format='gps', scale='utc').to_value('iso', subfmt='date_hms'))
df['t_end'] = df['t_end'].apply(lambda gps: Time(gps, format='gps', scale='utc').to_value('iso', subfmt='date_hms'))
# order by t_start
df = df.sort_values(by='t_start').reset_index(drop=True)
print(len(df.columns), df.head())
for col in df.columns:
    print(type(df[col][0]))
    if type(df[col][0]) not in ['list']:
        print(df[col].unique())
df.to_csv('/home/scutini/ACDAnomalies/acd/catalogs/gracedb_events_2024.csv')
table = Table.from_pandas(df)
table.write('/home/scutini/ACDAnomalies/acd/catalogs/gracedb_events_2024.fits', overwrite=True)