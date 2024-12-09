import requests

API_KEY = "8b9543b1f1c14ef888c3205e10dd6d27.mhePpzH7gmky2co3Hy0ZgTsN4kgl3vag"
BASE_URL = "https://cloud.lambdalabs.com/api/v1"

# Example: List all available instance types
headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get(f"{BASE_URL}/instance-types", headers=headers)

if response.status_code == 200:
    print("Instance Types:", response.json())
else:
    print("Error:", response.json())

