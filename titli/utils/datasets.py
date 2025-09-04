import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import csv

class StreamingCSVDataset(IterableDataset):
    def __init__(self, feature_csv_path, label_csv_path, max_samples=None, 
                 transform=None, label_column=0, skip_header=True):
        self.feature_csv_path = feature_csv_path
        self.label_csv_path = label_csv_path
        self.transform = transform
        self.max_samples = max_samples
        self.label_column = label_column  # Allow specifying which column contains labels
        self.skip_header = skip_header
        
        # Read headers to determine feature dimensions and validate structure
        with open(feature_csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            self.feature_headers = next(reader)
            self.input_size = len(self.feature_headers)
        
        # Check label file structure
        with open(label_csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
                raise ValueError(f"Label column index {self.label_column} is out of range. Valid indices are 0-{len(self.label_headers)-1}.")
    
    def _open_pair(self):
        """Open both CSV files and return file handles and readers"""
        # Open in text mode; newline='' for csv correctness
        f_feat = open(self.feature_csv_path, 'r', newline='')
        f_lab = open(self.label_csv_path, 'r', newline='')
        r_feat = csv.reader(f_feat)
        r_lab = csv.reader(f_lab)
        
        if self.skip_header:
            next(r_feat, None)
            next(r_lab, None)
            
        return f_feat, f_lab, r_feat, r_lab
    
    def __iter__(self):
        """Iterator that supports multi-worker data loading"""
        # Each worker gets its own file handles & shard using line-skipping
        worker = get_worker_info()
        f_feat, f_lab, r_feat, r_lab = self._open_pair()
        
        sample_count = 0
        
        if worker is not None:
            # Multi-worker setup: shard data by modulo to avoid pre-indexing
            worker_id = worker.id
            num_workers = worker.num_workers
            
            # Advance pointers until we hit our shard
            i = 0
            while True:
                try:
                    feat_row = next(r_feat)
                    lab_row = next(r_lab)
                except StopIteration:
                    break
                
                # Check if this sample belongs to current worker
                if (i % num_workers) == worker_id:
                    yield self._to_example(feat_row, lab_row)
                    sample_count += 1
                    
                    # Check max_samples limit
                    if self.max_samples and sample_count >= self.max_samples:
                        break
                        
                i += 1
        else:
            # Single-worker / no-workers
            for feat_row, lab_row in zip(r_feat, r_lab):
                yield self._to_example(feat_row, lab_row)
                sample_count += 1
                
                # Check max_samples limit
                if self.max_samples and sample_count >= self.max_samples:
                    break
        
        # Clean up file handles
        f_feat.close()
        f_lab.close()
    
    def _to_example(self, feat_row, lab_row):
        """Convert CSV rows to tensor example"""
        try:
            # Parse features
            x = torch.tensor([float(v) for v in feat_row], dtype=torch.float32)
            
            # Parse label using specified column
            if len(lab_row) > self.label_column:
                y = torch.tensor(float(lab_row[self.label_column]), dtype=torch.float32)
            else:
                # If label row is shorter than expected, raise an error
                raise ValueError(f"Label row is missing expected column {self.label_column}. Label row: {lab_row}")
                
        except ValueError as e:
            print(f"Error processing row: {e}")
            print(f"Feature row: {feat_row}")
            print(f"Label row: {lab_row}")
            raise
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


# Legacy StreamingCSVDataset for backward compatibility
class LegacyStreamingCSVDataset(Dataset):
    def __init__(self, feature_csv_path, label_csv_path, max_samples=None, transform=None, label_column=0):
        self.feature_csv_path = feature_csv_path
        self.label_csv_path = label_csv_path
        self.transform = transform
        self.max_samples = max_samples
        self.label_column = label_column  # Allow specifying which column contains labels
        
        # Get the total number of lines in the CSV (excluding header)
        self.total_samples = self._count_lines() - 1  # -1 for header
        
        if max_samples:
            self.total_samples = min(self.total_samples, max_samples)
        
        # Read headers to determine feature dimensions
        with open(feature_csv_path, 'r') as f:
            reader = csv.reader(f)
            self.feature_headers = next(reader)
            self.input_size = len(self.feature_headers)
        
        # Check label file structure
        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            self.label_headers = next(reader)
            # Check if the specified label column exists
            if self.label_column >= len(self.label_headers):
                raise ValueError(f"Label column index {self.label_column} not found. Label file has {len(self.label_headers)} columns.")
    
    def _count_lines(self):
        """Count total lines in the feature CSV file"""
        with open(self.feature_csv_path, 'r') as f:
            return sum(1 for _ in f)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError("Index out of range")
        
        # Read the specific line from feature CSV
        feature_row = self._read_line_from_csv(self.feature_csv_path, idx + 1)  # +1 to skip header
        label_row = self._read_line_from_csv(self.label_csv_path, idx)
        
        # Convert to tensors
        try:
            features = torch.tensor([float(x) for x in feature_row], dtype=torch.float32)
            # Use the specified label column (default is 0)
            if len(label_row) > self.label_column:
                label = torch.tensor(float(label_row[self.label_column]), dtype=torch.float32)
            else:
                raise ValueError(f"Missing label data at index {idx}: expected column {self.label_column}, got row {label_row}")
        except ValueError as e:
            print(f"Error processing row {idx}: {e}")
            print(f"Feature row: {feature_row}")
            print(f"Label row: {label_row}")
            raise
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def _read_line_from_csv(self, file_path, line_number):
        """Read a specific line number from CSV file"""
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == line_number:
                    return row
        raise IndexError(f"Line {line_number} not found in {file_path}")


# Usage example:
# from torch.utils.data import DataLoader
# 
# # Create streaming dataset
# ds = StreamingCSVDataset("features.csv", "labels.csv", label_column=0)
# loader = DataLoader(ds, batch_size=256, num_workers=4)  # no shuffle with IterableDataset
# 
# # Fast streaming training loop
# for xb, yb in loader:
#     # Process batch...
#     pass
