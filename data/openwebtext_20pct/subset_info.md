# openwebtext_20pct Dataset

This is a 20.0% subset of the original OpenWebText dataset.

## Dataset Information
- Source: data/openwebtext
- Subset percentage: 20.0%
- Creation method: First 20.0% of training tokens
- Validation data: Full validation set preserved

## Files
- `train.bin`: 20.0% of original training data
- `val.bin`: Full validation data
- `meta.pkl`: Vocabulary metadata (copied from original)
- `subset_info.md`: This information file

## Usage
Use this dataset by setting `dataset = 'openwebtext_20pct'` in your training configuration.

Original dataset info:
- Original training tokens: ~9B
- Original validation tokens: ~4M
- Total documents: 8,013,769
