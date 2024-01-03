import csv
import psycopg2
from datetime import datetime


def connect_db():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5433",
            database="postgres",
            user="postgres",
            password="W%2mN7&WkF")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


# Read data from CSV file
csv_file_path = "test.csv"
with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    data_from_csv = list(csv_reader)
# data = [[c.replace('\ufeff', '') for c in row] for row in data_from_csv]
print("Total rows: ", len(data_from_csv))
print("First row: ", data_from_csv[0])
#print first row id
print("First row id: ", data_from_csv[0]["id"])
# Connect to the database
conn = connect_db()

# Insert data into the "contents" table

cursor = conn.cursor()
query = """
        INSERT INTO smcc.contents (
            "id", "sourceId", "topicIds", "link", "type", "textContent",
            "title", "imageContents", "videoContents", "likes", "shares", "comments",
            "views", "totalReactions", "reactionsPerHour", "commentIds", "status",
            "category", "postedAt", "process", "screenShot",
            "editedTextContent", "violationContent", "violationEnactment", "violationTimes", "userHandle", "blockRequire", "viettelBlocked",
           "fptBlocked", "vnptBlocked", "contentTeleNoti", "createdAt",
           "updatedAt"
        ) VALUES (
            %(id)s, %(sourceId)s, %(topicIds)s, %(link)s, %(type)s, %(textContent)s,
            %(title)s, %(imageContents)s, %(videoContents)s, %(likes)s, %(shares)s, %(comments)s,
            %(views)s, %(totalReactions)s, %(reactionsPerHour)s, %(commentIds)s, %(status)s,
            %(category)s, %(postedAt)s, %(process)s, %(screenShot)s,
            %(editedTextContent)s, %(violationContent)s, %(violationEnactment)s, %(violationTimes)s, %(userHandle)s, %(blockRequire)s, %(viettelBlocked)s,
            %(fptBlocked)s, %(vnptBlocked)s, %(contentTeleNoti)s, NOW(), NOW()
        )
        """


# Execute the query for each row in the CSV file
for row in data_from_csv:
    # print(row)

    # Execute the query
    cursor.execute(query, row)
    conn.commit()
    # print("Inserted row: ", row["id"])
# insert_into_contents(conn, data_from_csv)
print("Inserted successfully")
# Close the database connection
conn.close()
