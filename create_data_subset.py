#!/usr/bin/env python3
"""
Create a subset of the OpenWebText dataset for data analysis experiments.

This script creates a new dataset directory with a specified percentage of the 
original training data, maintaining the same structure and format.

Usage:
    python create_data_subset.py --subset_pct 20.0 --subset_name openwebtext_20pct
"""

import os
import argparse
import numpy as np
import shutil
from pathlib import Path


def create_data_subset(source_dir, target_dir, subset_pct=20.0, preserve_val=True):
    """
    Create a subset of the dataset by taking the first subset_pct% of training data.
    
    Args:
        source_dir: Path to source dataset directory (e.g., 'data/openwebtext')
        target_dir: Path to target subset directory (e.g., 'data/openwebtext_20pct') 
        subset_pct: Percentage of training data to include (default: 20.0)
        preserve_val: Whether to copy the full validation set (default: True)
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {subset_pct}% subset of {source_dir} in {target_dir}")
    
    # Process training data
    train_source = source_path / 'train.bin'
    train_target = target_path / 'train.bin'
    
    if train_source.exists():
        print(f"Processing training data: {train_source}")
        
        # Load the original training data
        original_data = np.memmap(train_source, dtype=np.uint16, mode='r')
        original_size = len(original_data)
        
        # Calculate subset size
        subset_size = int(original_size * (subset_pct / 100.0))
        
        print(f"Original training data: {original_size:,} tokens")
        print(f"Subset size: {subset_size:,} tokens ({subset_size/original_size*100:.1f}%)")
        
        # Create subset by taking first subset_pct% of data
        subset_data = original_data[:subset_size]
        
        # Save subset to new file
        print(f"Saving subset to: {train_target}")
        with open(train_target, 'wb') as f:
            subset_data.tofile(f)
        
        print(f"Training subset created: {subset_size:,} tokens")
        
        # Verify the saved data
        saved_data = np.memmap(train_target, dtype=np.uint16, mode='r')
        print(f"Verification: saved {len(saved_data):,} tokens")
        
    else:
        print(f"Warning: {train_source} not found")
    
    # Process validation data
    val_source = source_path / 'val.bin'
    val_target = target_path / 'val.bin'
    
    if val_source.exists() and preserve_val:
        print(f"Copying full validation data: {val_source} -> {val_target}")
        shutil.copy2(val_source, val_target)
        
        # Verify validation data
        val_data = np.memmap(val_target, dtype=np.uint16, mode='r')
        print(f"Validation data: {len(val_data):,} tokens")
        
    elif val_source.exists():
        # If not preserving full validation set, create a subset
        print(f"Creating validation subset: {val_source}")
        
        original_val = np.memmap(val_source, dtype=np.uint16, mode='r')
        val_subset_size = int(len(original_val) * (subset_pct / 100.0))
        val_subset = original_val[:val_subset_size]
        
        with open(val_target, 'wb') as f:
            val_subset.tofile(f)
            
        print(f"Validation subset created: {val_subset_size:,} tokens")
    else:
        print(f"Warning: {val_source} not found")
    
    # Copy metadata files if they exist
    metadata_files = ['meta.pkl', 'readme.md', 'prepare.py']
    for meta_file in metadata_files:
        source_meta = source_path / meta_file
        target_meta = target_path / meta_file
        
        if source_meta.exists():
            print(f"Copying metadata: {meta_file}")
            shutil.copy2(source_meta, target_meta)
    
    # Create a new readme for the subset
    subset_readme = target_path / 'subset_info.md'
    with open(subset_readme, 'w') as f:
        f.write(f"""# {target_path.name} Dataset

This is a {subset_pct}% subset of the original OpenWebText dataset.

## Dataset Information
- Source: {source_dir}
- Subset percentage: {subset_pct}%
- Creation method: First {subset_pct}% of training tokens
- Validation data: {'Full validation set preserved' if preserve_val else f'{subset_pct}% subset'}

## Files
- `train.bin`: {subset_pct}% of original training data
- `val.bin`: {'Full' if preserve_val else f'{subset_pct}% of'} validation data
- `meta.pkl`: Vocabulary metadata (copied from original)
- `subset_info.md`: This information file

## Usage
Use this dataset by setting `dataset = '{target_path.name}'` in your training configuration.

Original dataset info:
- Original training tokens: ~9B
- Original validation tokens: ~4M
- Total documents: 8,013,769
""")
    
    print(f"\nDataset subset creation complete!")
    print(f"New dataset location: {target_path}")
    print(f"To use this dataset, set: dataset = '{target_path.name}'")


def main():
    parser = argparse.ArgumentParser(description="Create a subset of OpenWebText dataset")
    parser.add_argument('--source_dir', type=str, default='data/openwebtext',
                       help='Source dataset directory (default: data/openwebtext)')
    parser.add_argument('--target_dir', type=str, default=None,
                       help='Target subset directory (auto-generated if not specified)')
    parser.add_argument('--subset_pct', type=float, default=20.0,
                       help='Percentage of training data to include (default: 20.0)')
    parser.add_argument('--subset_name', type=str, default=None,
                       help='Name for the subset (default: original_name_XXpct)')
    parser.add_argument('--preserve_full_val', action='store_true', default=True,
                       help='Preserve full validation set (default: True)')
    
    args = parser.parse_args()
    
    # Auto-generate target directory if not specified
    if args.target_dir is None:
        if args.subset_name:
            args.target_dir = f"data/{args.subset_name}"
        else:
            source_name = Path(args.source_dir).name
            args.target_dir = f"data/{source_name}_{int(args.subset_pct)}pct"
    
    # Create the subset
    create_data_subset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        subset_pct=args.subset_pct,
        preserve_val=args.preserve_full_val
    )


if __name__ == "__main__":
    main()
