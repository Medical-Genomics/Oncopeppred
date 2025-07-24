import requests

response = requests.get("http://127.0.0.1:5000/peptides")
print(response.json())  # Print the returned JSON data
