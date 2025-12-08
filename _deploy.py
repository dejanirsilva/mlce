#!/usr/bin/env python3
"""
Copy files from _site to root for GitHub Pages deployment
Also copy site_libs to Module directories for slide compatibility
"""
import shutil
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent
site_dir = project_root / "_site"

# Files and directories to copy to root
files_to_copy = [
    ("index.html", "index.html"),
    ("site_libs", "site_libs"),
    ("search.json", "search.json"),
]

print("Copying files from _site/ to root directory...")

for source, dest in files_to_copy:
    source_path = site_dir / source
    dest_path = project_root / dest
    
    if source_path.exists():
        if source_path.is_dir():
            # Remove existing directory if it exists
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
            print(f"  Copied directory: {source} -> {dest}")
        else:
            shutil.copy2(source_path, dest_path)
            print(f"  Copied file: {source} -> {dest}")
    else:
        print(f"  Skipped (not found): {source}")

# Also copy site_libs to Module directories for slide compatibility
print("\nCopying site_libs to Module directories for slide compatibility...")
module_dirs = ["Module01", "Module02", "Module03", "Module04", "Module05"]
site_libs_root = project_root / "site_libs"

if site_libs_root.exists():
    for module_dir in module_dirs:
        module_path = project_root / module_dir
        if module_path.exists():
            module_site_libs = module_path / "site_libs"
            if module_site_libs.exists():
                shutil.rmtree(module_site_libs)
            shutil.copytree(site_libs_root, module_site_libs)
            print(f"  Copied site_libs to {module_dir}/")

print("\nDeployment files copied successfully!")
