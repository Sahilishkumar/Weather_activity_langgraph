import os, requests
from dotenv import load_dotenv
load_dotenv()
res = requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f"Bearer {os.environ.get('GROQ_API_KEY')}"})
print(res.json())
