from sqlite3 import Connection, connect
from fastapi import Request

db_location: str = "db/freq_analytics.sqlite"


def check_and_create_db(db_location: str = db_location) -> Connection:
    """Checks if the database exists. If the database does not exist, it will be created.
    Arguments:
        db_location: The location of the SQLite database file.
    Returns:
        sqlite3.Connection: A connection object for the SQLite database.
    """

    connection = connect(db_location)
    cursor = connection.cursor()

    check_query: str = """ SELECT name 
            FROM sqlite_master 
            WHERE type='table' 
            AND name='analytics';"""

    if cursor.execute(check_query).fetchone() is None:
        create_query: str = """CREATE TABLE analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_agent TEXT,
            method TEXT,
            path TEXT,
            best_station TEXT,
            station_name TEXT,
            verbatims TEXT,
            tags TEXT,
            artists TEXT
        );"""
        cursor.execute(create_query)
        connection.commit()

    return connection


def log_analytics(
    request: Request,
    best_station: str,
    station_name: str,
    verbatims: list[str],
    tags: list[str],
    artists: list[str],
    db_location: str = db_location,
) -> None:
    """Logs analytics data to the SQLite database.
    Arguments:
        request: The FastAPI request object.
        best_station: The best station determined by the model.
        station_name: The name of the generated station.
        verbatims: The list of verbatims associated with the request.
        tags: The list of tags associated with the request.
        artists: The list of artists associated with the request.
        db_location: The location of the SQLite database file.
    """
    connection = check_and_create_db(db_location)
    cursor = connection.cursor()

    insert_query = """INSERT INTO analytics 
        (user_agent, method, path, best_station, station_name, verbatims, tags, artists) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""

    data: tuple = (
        request.headers.get("user-agent"),
        request.method,
        str(request.url.path),
        best_station,
        station_name,
        ",".join(verbatims),
        ",".join(tags),
        ",".join(artists),
    )

    cursor.execute(insert_query, data)

    connection.commit()
    connection.close()


def get_count_questionnaires(db_location: str = db_location) -> int:
    """Retrieves all analytics data from the SQLite database.
    Arguments:
        db_location: The location of the SQLite database file.
    Returns:
        int: Count of all completed 
    """
    connection = check_and_create_db(db_location)
    cursor = connection.cursor()
    select_query = """SELECT COUNT(*) FROM analytics ORDER BY timestamp DESC;"""
    cursor.execute(select_query)
    rows = cursor.fetchall()
    connection.close()
    return rows[0][0] if rows else 0