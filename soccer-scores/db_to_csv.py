import csv
import sqlite3


def create_csv(csv_filename="file.csv", db_filename="database.sqlite"):
    conn = sqlite3.connect(db_filename)  # Replace with your database file
    cursor = conn.cursor()

    # Calculate the 'outcome' based on the goals
    cursor.execute("""
    SELECT substr(date, 1, 10) as date, home_team_api_id, away_team_api_id,
    CASE
        WHEN home_team_goal > away_team_goal THEN 'H'
        WHEN home_team_goal < away_team_goal THEN 'A'
        ELSE 'D'
    END as outcome
    FROM Match
    """)

    rows = cursor.fetchall()

    # Define the column headers
    headers = ["date", "home_team_api_id", "away_team_api_id", "outcome"]

    # Write to a CSV file
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    conn.close()

    print("CSV file with outcomes created successfully!")


if __name__ == "__main__":
    create_csv()
