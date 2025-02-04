import sqlite3
import os

def read_chat_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: The database file '{db_path}' does not exist.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get the list of tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
            return

        print("Tables in the database:")
        for table in tables:
            print(f"- {table[0]}")

        # Assuming there's a table named 'conversations'
        # Modify this query based on the actual structure of your database
        cursor.execute("SELECT * FROM conversations LIMIT 10")
        conversations = cursor.fetchall()

        if not conversations:
            print("No conversations found in the database.")
        else:
            print("\nFirst 10 conversations:")
            for conversation in conversations:
                print(conversation)

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    db_path = os.path.expanduser("~/Library/Messages/chat.db")
    read_chat_db(db_path)
