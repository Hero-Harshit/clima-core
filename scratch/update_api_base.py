import glob

target = 'const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";'
replacement = 'const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");'

count = 0
for path in glob.glob('frontend/src/**/*.ts*', recursive=True):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if target in content:
        content = content.replace(target, replacement)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1
        print(f'Updated {path}')

print(f'Total updated: {count}')
