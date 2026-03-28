import re

with open('inference.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the debug prints and temperature back to a stable 0.5
text = re.sub(
    r'        # 2\. Extract Features\s+descriptors, color_hist = get_orb_and_color_features\(img\)\s+# 3\. BoVW Histogram',
    '        # 2. Extract Features\n        descriptors, color_hist = get_orb_and_color_features(img)\n\n        # Reject if not enough features (background/blank/blur)\n        if len(descriptors) < 50:\n            return {"label": "none", "confidence": 0.0}\n\n        # 3. BoVW Histogram',
    text
)

# And fix temperature to scale the decision function to sensible values (0.35 works well for giving ~0.9+ for true matches)
# Wait, it's already 0.35 in my previous patch

with open('inference.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Patched inference.py with descriptor check.')
