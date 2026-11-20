import urllib.request
import json

url = 'https://zenodo.org/api/records/14999580'
req = urllib.request.Request(url, headers={'User-Agent': 'ClimaCore/2.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode())
        meta = data.get('metadata', {})
        print('Title:', meta.get('title'))
        print('Version:', meta.get('version'))
        print('Publication date:', meta.get('publication_date'))
        print('License:', meta.get('license'))
        print('Description excerpt:', meta.get('description', '')[:300])
        print('\nFiles:')
        for f in data.get('files', []):
            size_mb = round(f.get('size', 0) / (1024*1024), 2)
            key = f.get('key')
            link = f.get('links', {}).get('self')
            checksum = f.get('checksum')
            print(f"  {key:50s} | {size_mb:8.2f} MB | {checksum} | {link}")
except Exception as e:
    print('Error accessing Zenodo API:', e)
