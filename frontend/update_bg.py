import os, glob, re

dir_path = r'd:\federated_learning pipeline_rul\frontend\src\pages'
for fp in glob.glob(os.path.join(dir_path, '*.tsx')):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove gradient background
    content = content.replace(' bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#0a162c] via-background to-background', '')
    
    # Replace standard card backgrounds
    content = re.sub(r'bg-\[\#050B14\]/\d+\s+backdrop-blur-md\s+border\s+border-primary/\d+', 'glass-panel', content)
    
    # Edge cases
    content = re.sub(r'bg-\[\#050B14\]/\d+\s+backdrop-blur-md\s+border\s+rounded-xl', 'glass-panel rounded-xl', content)
    content = re.sub(r'bg-\[\#050B14\]/\d+\s+backdrop-blur-md\s+border\s+border-l-4', 'glass-panel border-l-4', content)
    
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(content)

print('Done!')
