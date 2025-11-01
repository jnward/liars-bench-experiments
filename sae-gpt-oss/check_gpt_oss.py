#!/usr/bin/env python3
"""Quick script to check if GPT-OSS-20B is available on Neuronpedia."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
NEURONPEDIA_API_KEY = os.getenv('NEURONPEDIA_API_KEY')
BASE_URL = 'https://www.neuronpedia.org/api'

def test_model(model_id, layer_pattern):
    url = f'{BASE_URL}/feature/{model_id}/{layer_pattern}/0'
    headers = {'Content-Type': 'application/json', 'x-api-key': NEURONPEDIA_API_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, response.status_code
    except Exception as e:
        return False, str(e)

print('🔍 Checking GPT-OSS-20B availability...\n')

# Most likely naming patterns
test_cases = [
    ('gpt-oss-20b', '0-res-canonical'),
    ('gpt-oss-20b', '0-res'),
    ('gpt-oss-20b', '0-res-jb'),
]

for model_id, layer_pattern in test_cases:
    success, result = test_model(model_id, layer_pattern)
    if success:
        print(f'✓✓✓ SUCCESS! ✓✓✓')
        print(f'Model: {model_id}')
        print(f'Layer: {layer_pattern}')
        print(f'Available fields: {list(result.keys())[:10]}')
        exit(0)
    else:
        print(f'✗ {model_id}/{layer_pattern}: {result}')

print(f'\n❌ GPT-OSS-20B not yet available')
