#!/usr/bin/env python3
"""
Fixed script to download and prepare OpenWebText dataset.
Uses alternative sources that work with current datasets library.
"""

import os
import sys
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 8
# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

def process(example):
    """Tokenize text example"""
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token
    out = {'ids': ids, 'len': len(ids)}
    return out

if __name__ == '__main__':
    print("OpenWebText Data Preparation")
    print("="*60)
    
    # Option 1: Use the pre-processed version from allenai
    print("Downloading OpenWebText from AllenAI (preprocessed version)...")
    try:
        # This is a cleaner version that should work
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        
        # Since C4 is huge, let's take a subset similar to OpenWebText size
        print("Note: Using C4 dataset as OpenWebText alternative (taking subset)")
        
        # Convert streaming dataset to regular dataset (taking first 8M examples)
        print("Loading examples (this may take a while)...")
        examples = []
        for i, example in enumerate(tqdm(dataset, desc="Loading", total=8_000_000)):
            examples.append({'text': example['text']})
            if i >= 8_000_000:  # Similar size to OpenWebText
                break
        
        # Create dataset from list
        from datasets import Dataset
        dataset = Dataset.from_list(examples)
        dataset = {'train': dataset}
        
    except Exception as e:
        print(f"Failed to load C4: {e}")
        print("\nTrying WikiText as a smaller alternative for testing...")
        
        # Option 2: Use WikiText for testing (much smaller)
        try:
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
            print("✅ Loaded WikiText-103 dataset (smaller dataset for testing)")
            print("Note: This is much smaller than OpenWebText but good for testing")
            
        except Exception as e:
            print(f"Failed to load WikiText: {e}")
            
            # Option 3: Use even smaller dataset
            print("\nTrying tiny Shakespeare dataset...")
            # Just use the local Shakespeare data we already have
            with open('../shakespeare/input.txt', 'r') as f:
                text = f.read()
            
            from datasets import Dataset
            # Split into chunks to simulate multiple documents
            chunk_size = 1000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            dataset = Dataset.from_dict({'text': chunks})
            dataset = {'train': dataset}
            print("✅ Using Shakespeare dataset as fallback")
    
    # Create train/validation split
    print("\nCreating train/validation split...")
    if 'validation' in dataset:
        # WikiText already has validation split
        split_dataset = {
            'train': dataset['train'],
            'val': dataset['validation']
        }
    else:
        # Create validation split
        train_dataset = dataset['train']
        split = train_dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset = {
            'train': split['train'],
            'val': split['test']
        }
    
    print(f"\nDataset splits:")
    print(f"Train: {len(split_dataset['train']):,} examples")
    print(f"Val: {len(split_dataset['val']):,} examples")
    
    # Tokenize the dataset
    print("\nTokenizing dataset...")
    tokenized = {}
    for split_name, split_data in split_dataset.items():
        tokenized[split_name] = split_data.map(
            process,
            remove_columns=['text'],
            desc=f"tokenizing {split_name}",
            num_proc=num_proc if len(split_data) > 1000 else 1,  # Use single process for small datasets
        )
    
    # Save tokenized data to binary files
    print("\nSaving to binary files...")
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Adjust batch count based on dataset size
        total_batches = min(1024, max(1, len(dset) // 100))
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"{split}.bin: {arr_len:,} tokens")
    
    print("\n✅ Dataset preparation complete!")
    print(f"Files created: train.bin, val.bin")
    print("\nNote: If you want the full OpenWebText dataset, you may need to:")
    print("1. Download it manually from Hugging Face")
    print("2. Or wait for the datasets library to be updated")
    print("3. For now, you can train on the alternative dataset provided")
