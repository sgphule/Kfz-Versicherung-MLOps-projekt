import config
import pandas as pd
import pymongo
import logging
logging.basicConfig(level=logging.INFO)

def main():
    try:
        df = pd.read_csv(config.SOURCE_FILE)
        if df.empty:
            logging.warning("CSV file is empty.")
            return
    except FileNotFoundError:
        logging.debug("Source file not found.")
        return
    except pd.errors.ParserError:
        logging.debug("Error parsing CSV.")
        return
    data = df.to_dict(orient="records")

    with pymongo.MongoClient(config.CONNECTION_URL) as client:
        client.admin.command('ping')
        db = client[config.DB_NAME]
        collection = db[config.COLLECTION_NAME]
        if not data:
            logging.debug("No data to insert.")
            return
        result = collection.insert_many(data)
        logging.info(f"Inserted {len(result.inserted_ids)} records.")


if __name__ == "__main__":
    main()
