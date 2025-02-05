from dotenv import load_dotenv
import os
import psycopg2

conn = None

def __init__():
    global conn
    # Load environment variable from .env file
    load_dotenv()

    # PostgreSQL Connection
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB_NAME"),
        user=os.getenv("POSTGRES_DB_USER"),
        password=os.getenv("POSTGRES_DB_PASSWORD"),
        host=os.getenv("POSTGRES_DB_HOST"),
        port=os.getenv("POSTGRES_DB_PORT")
    )

    # print("\nInitialize the db connection\n")

def get_db():
    cursor = conn.cursor()
    return cursor

def close_db():
    # Commit and close the connection
    conn.commit()
    conn.close()

# Check if the script is being run directly
if __name__ == "__main__":
    __init__()