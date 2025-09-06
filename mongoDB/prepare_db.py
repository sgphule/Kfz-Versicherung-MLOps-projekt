import pandas as pd
import pymongo

from mongoDB import config

df = pd.read_csv(config.SOURCE_FILE)

data = df.to_dict(orient="records")

DB_NAME = config.DB_NAME
COLLECTION_NAME = config.COLLECTION_NAME
CONNECTION_URL = config.CONNECTION_URL

client = pymongo.MongoClient(CONNECTION_URL)
database = client[DB_NAME]
collection = database[COLLECTION_NAME]

rec = collection.insert_many(data)