import requests
import json

response = requests.get("https://data.enedis.fr/explore/dataset/bilan-electrique-demi-heure/api/?sort=horodate&dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6ImJpbGFuLWVsZWN0cmlxdWUtZGVtaS1oZXVyZSIsIm9wdGlvbnMiOnsic29ydCI6Imhvcm9kYXRlIn19LCJjaGFydHMiOlt7ImFsaWduTW9udGgiOnRydWUsInR5cGUiOiJsaW5lIiwiZnVuYyI6IkFWRyIsInlBeGlzIjoiaW5qZWN0aW9uX3J0ZSIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiMwMDVFQjgifSx7ImFsaWduTW9udGgiOnRydWUsInR5cGUiOiJsaW5lIiwiZnVuYyI6IkFWRyIsInlBeGlzIjoiY29uc29tbWF0aW9uX3RvdGFsZSIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiMwMEEzRTAifV0sInhBeGlzIjoiaG9yb2RhdGUiLCJtYXhwb2ludHMiOjIwMCwidGltZXNjYWxlIjoiZGF5Iiwic29ydCI6IiJ9XSwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZSwic2luZ2xlQXhpcyI6dHJ1ZX0%3D&rows=60000")
print(response.status_code)