import requests
import os

url = "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/master/dist/northwind.db"
path = "data/northwind.db"

def download_db():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading Northwind database...")
        response = requests.get(url)

        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            raise Exception("Failed to download DB")
    else:
        print("Northwind database already exists.")

