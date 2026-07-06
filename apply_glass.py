import os, re

pages_dir = r"d:\federated_learning pipeline_rul\frontend\src\pages"

# Pattern: any combination of bg-[#050B14]/XX + backdrop-blur + border + border-primary/XX inline Tailwind
# We'll replace these inline combos on container divs with the glass-panel class
patterns_to_remove = [
    r'bg-\[#050B14\]/(?:60|80)\s+backdrop-blur-md\s+border\s+border-primary/20',
    r'bg-\[#050B14\]/(?:60|80)\s+backdrop-blur-md\s+border\s+border-primary/(?:10|15|20|25|30)',
    r'bg-\[#02050A\]\s+backdrop-blur-md\s+border\s+border-primary/(?:10|15|20|25|30)',
    r'bg-\[#02050A\]/(?:60|80|90)\s+backdrop-blur-md\s+border\s+border-primary/(?:10|15|20|25|30)',
]

for fname in os.listdir(pages_dir):
    if not fname.endswith('.tsx'):
        continue
    fpath = os.path.join(pages_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for pat in patterns_to_remove:
        content = re.sub(pat, 'glass-panel', content)
    
    if content != original:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {fname}")
    else:
        print(f"No changes: {fname}")

print("Done.")
