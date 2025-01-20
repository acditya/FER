import requests

url = 'http://20.46.48.174:5000/upload'
file_path = 'Adi_DP.jpg'

with open(file_path, 'rb') as file:
    files = {'file': file}
    response = requests.post(url, files=files)

print(response.json())