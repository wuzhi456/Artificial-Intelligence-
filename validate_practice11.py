#!/usr/bin/env python3
"""
Validation script for Practice11 SVM implementation
This script validates the notebook structure and code without requiring the dataset
"""

import json
import re
import sys

def validate_notebook():
    """Validate the Practice11 notebook structure and implementation"""
    
    print("=" * 80)
    print("Practice11 Notebook Validation")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    # Load notebook
    try:
        with open("Practice11 - Credit Card Fraud Detection using SVM.ipynb", 'r') as f:
            nb = json.load(f)
        print("✓ Notebook loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load notebook: {e}")
        return False
    
    # Check notebook structure
    if 'cells' not in nb:
        errors.append("Notebook missing 'cells' key")
        return False
    
    print(f"✓ Notebook has {len(nb['cells'])} cells")
    
    # Check for required imports
    all_code = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            all_code.append(''.join(cell.get('source', [])))
    
    full_code = '\n'.join(all_code)
    
    required_imports = {
        'SVC': 'from sklearn.svm import SVC',
        'LogisticRegression': 'from sklearn.linear_model import LogisticRegression',
        'time': 'import time',
        'KFold': 'KFold',
        'confusion_matrix': 'confusion_matrix',
    }
    
    print("\nChecking required imports:")
    for name, pattern in required_imports.items():
        if pattern in full_code:
            print(f"  ✓ {name} imported")
        else:
            errors.append(f"Missing import: {name}")
            print(f"  ✗ {name} NOT found")
    
    # Check for SVM-specific functions
    print("\nChecking SVM implementation:")
    
    if 'printing_Kfold_scores_svm' in full_code:
        print("  ✓ SVM K-fold function defined")
    else:
        errors.append("SVM K-fold function not found")
        print("  ✗ SVM K-fold function NOT found")
    
    if 'best_c_svm' in full_code and 'best_kernel_svm' in full_code:
        print("  ✓ SVM parameter selection implemented")
    else:
        errors.append("SVM parameter variables not found")
        print("  ✗ SVM parameter selection NOT found")
    
    # Check for kernel options
    if "kernel_options = ['linear', 'rbf']" in full_code or \
       'kernel_options = ["linear", "rbf"]' in full_code:
        print("  ✓ Multiple kernel options defined")
    else:
        warnings.append("Kernel options might not be properly defined")
        print("  ⚠ Kernel options might not be defined")
    
    # Check for timing measurements
    if 'time.time()' in full_code and 'training_time' in full_code:
        print("  ✓ Timing measurements implemented")
    else:
        warnings.append("Timing measurements not found")
        print("  ⚠ Timing measurements NOT found")
    
    # Check that SMOTE section is commented/handled
    print("\nChecking SMOTE handling:")
    smote_commented = False
    for cell in nb['cells']:
        source = ''.join(cell.get('source', []))
        if 'SMOTE' in source and 'COMMENTED OUT' in source:
            smote_commented = True
            break
    
    if smote_commented:
        print("  ✓ SMOTE section properly handled (commented)")
    else:
        warnings.append("SMOTE section handling unclear")
        print("  ⚠ SMOTE section might need attention")
    
    # Check for comparison functionality
    print("\nChecking comparison features:")
    
    if 'printing_Kfold_scores_lr' in full_code:
        print("  ✓ LR function kept for comparison")
    else:
        warnings.append("LR comparison function not found")
        print("  ⚠ LR comparison function NOT found")
    
    # Count SVM model instances
    svm_instances = full_code.count('SVC(')
    if svm_instances >= 3:
        print(f"  ✓ SVM model instantiated {svm_instances} times")
    else:
        warnings.append(f"Only {svm_instances} SVM instances found")
        print(f"  ⚠ Only {svm_instances} SVM instances found")
    
    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    if errors:
        print(f"\n✗ {len(errors)} ERROR(S) FOUND:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("\n✓ No critical errors found")
    
    if warnings:
        print(f"\n⚠ {len(warnings)} WARNING(S):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    if not errors and not warnings:
        print("\n✓✓✓ All checks passed! Notebook is ready for execution.")
        return True
    elif not errors:
        print("\n✓ Notebook structure is valid (with minor warnings)")
        return True
    else:
        print("\n✗ Notebook has critical errors that need to be fixed")
        return False

if __name__ == "__main__":
    success = validate_notebook()
    sys.exit(0 if success else 1)
