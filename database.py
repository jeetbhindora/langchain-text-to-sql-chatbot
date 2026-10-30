from dotenv import dotenv_values
from sqlalchemy import URL, create_engine, text
from langchain_community.utilities import SQLDatabase

# Load configuration
config = dotenv_values(".env")

required_keys = [
    "DB_USER",
    "DB_PASSWORD",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_READONLY_USER",
    "DB_READONLY_PASSWORD",
]

for key in required_keys:
    if not config.get(key):
        raise ValueError(f"Missing {key} in .env file")


# Create connection URLs
db_url = URL.create(
    drivername="postgresql+psycopg",
    username=config["DB_USER"],
    password=config["DB_PASSWORD"],
    host=config["DB_HOST"],
    port=int(config["DB_PORT"]),
    database=config["DB_NAME"],
)

readonly_db_url = URL.create(
    drivername="postgresql+psycopg",
    username=config["DB_READONLY_USER"],
    password=config["DB_READONLY_PASSWORD"],
    host=config["DB_HOST"],
    port=int(config["DB_PORT"]),
    database=config["DB_NAME"],
)


# Create database engines
engine = create_engine(
    db_url,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=10000",
    },
)

read_engine = create_engine(
    readonly_db_url,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=10000",
    },
)


# Create LangChain database interface
db = SQLDatabase(
    read_engine,
    include_tables=["customers", "products", "sales"],
    sample_rows_in_table_info=2,
    view_support=False,
)


def test_connections():
    with engine.connect() as connection:
        print("Admin connection:", connection.execute(
            text("SELECT current_user")
        ).scalar())

    with read_engine.connect() as connection:
        print("Read-only connection:", connection.execute(
            text("SELECT current_user")
        ).scalar())

    print("Available tables:", db.get_usable_table_names())


if __name__ == "__main__":
    test_connections()