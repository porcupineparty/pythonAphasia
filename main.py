import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
# Define the API endpoint URL
API_URL = "https://api.openai.com/v1/completions"
MY_KEY = os.getenv("API_KEY")
# Headers containing your API key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {MY_KEY}",
    
}

# Sample requests to submit
requests_data = [
    {
        "prompt": "Translate 'Hello, how are you?' to French.",
        "model": "gpt-3.5-turbo-instruct",
        "temperature": 0.7,
        "max_tokens": 50
    },
    {
        "prompt": "Summarize the following paragraph: My chicken is a good chicken.",
        "model": "gpt-3.5-turbo-instruct",
        "temperature": 0.7,
        "max_tokens": 50
    }
    # Add more requests as needed
]

# Function to make requests to the API
def make_api_request(data):
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()

# Loop through each request data and make API calls
for request_data in requests_data:
    response = make_api_request(request_data)
    print("Request:", request_data)
    print("Response:", response)
    print("\n")
