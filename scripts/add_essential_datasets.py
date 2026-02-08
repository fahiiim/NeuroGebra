"""
Script to add essential sklearn-based datasets to the base Datasets class.

This copies methods from ExpandedDatasets to ensure base datasets are available.
"""

# Read the expanded loaders file and extract specific methods
essential_methods = [
    'load_wine',
    'load_breast_cancer', 
    'load_digits',
    'load_california_housing',
    'load_diabetes',
    'load_spam',
    'load_titanic',
    'load_credit_default',
    'load_mushroom',
    'load_adult_income',
    'load_boston_housing',
    'load_auto_mpg',
    'load_concrete_strength',
    'load_bike_sharing',
]

# Read expanded_loaders.py
with open('src/neurogebra/datasets/expanded_loaders.py', 'r', encoding='utf-8') as f:
    expanded_content = f.read()

# Extract each method
import re

methods_code = []
for method_name in essential_methods:
    # Find the method in the file
    pattern = rf'(@staticmethod\s+def {method_name}\(.*?\n(?:.*?\n)*?        return .*?\n\n)'
    match = re.search(pattern, expanded_content, re.MULTILINE | re.DOTALL)
    
    if match:
        method_code = match.group(1).strip()
        methods_code.append(f"\n    {method_code}\n")
        print(f"✓ Found {method_name}")
    else:
        print(f"✗ Could not find {method_name}")

# Read current loaders.py
with open('src/neurogebra/datasets/loaders.py', 'r', encoding='utf-8') as f:
    loaders_content = f.read()

# Add methods before the utility methods section
insertion_point = loaders_content.find('    # ============================================\n    # UTILITY METHODS')

if insertion_point == -1:
    print("ERROR: Could not find insertion point")
else:
    # Insert the methods
    new_section = '\n    # ============================================\n    # SKLEARN-BASED DATASETS\n    # ============================================\n'
    new_section += '\n'.join(methods_code)
    new_section += '\n'
    
    new_content = loaders_content[:insertion_point] + new_section + loaders_content[insertion_point:]
    
    # Write back
    with open('src/neurogebra/datasets/loaders.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"\n✅ Added {len(methods_code)} methods to loaders.py")
    print("📝 File updated successfully!")
