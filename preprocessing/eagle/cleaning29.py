
import pandas as pd
from pathlib import Path
import zipfile

file_path = Path("data") / "esif.hpc.eagle.ganglia.gpu.sixnodes_r104u29.csv.zip"
filename = 'esif.hpc.eagle.ganglia.gpu.sixnodes_r104u29.csv'
parquet_output = Path("filtered_node_r103u17.parquet")

# Columns we care about based on your analysis
target_cols = ['__time', 'dv', 'mt', 'vl']
chunk_size = 5_000_000

print("== STEP 1: Peeking at the first 1M rows ==")
with zipfile.ZipFile(file_path, 'r') as z:
    with z.open(filename, 'r') as f:
        df_peek = pd.read_csv(
            f, 
            encoding='latin-1', 
            usecols=target_cols, 
            nrows=1_000_000
        )

print(f"Peek Shape: {df_peek.shape[0]} rows, {df_peek.shape[1]} cols")
print("Head:")
print(df_peek.head())
print("\nUnique Metrics (mt):")
print(df_peek['mt'].unique())
print(f"\nTime Range: min = {df_peek['__time'].min()}, max = {df_peek['__time'].max()}")
print("-" * 50)


print("== STEP 2 & 3: Processing in chunks & filtering ==")
# Define the specific metrics you want to extract
desired_metrics = [
    'gpu0_temp', 'gpu1_temp', 
    'gpu0_power_usage_report', 'gpu1_power_usage_report', 
    'gpu0_mem_util', 'gpu1_mem_util'
] # Adjust metric names precisely matching your .unique() output above

filtered_chunks = []

with zipfile.ZipFile(file_path, 'r') as z:
    with z.open(filename, 'r') as f:
        # Read the file in chunks using the file stream (no extraction to disk!)
        for chunk_idx, chunk in enumerate(pd.read_csv(
            f, 
            encoding='latin-1', 
            usecols=target_cols, 
            chunksize=chunk_size,
            low_memory=False
        )):
            # Filter down rows to only the metrics we want
            # (Check if your exact metric names match string patterns or exact matches)
            mask = chunk['mt'].isin(desired_metrics)
            sub_chunk = chunk[mask].copy()
            
            if not sub_chunk.empty:
                # Convert value to float32 to save memory
                sub_chunk['vl'] = pd.to_numeric(sub_chunk['vl'], errors='coerce').astype('float32')
                
                # Convert string columns to category after filtering
                sub_chunk['mt'] = sub_chunk['mt'].astype('category')
                sub_chunk['dv'] = sub_chunk['dv'].astype('category')
                
                # Parse timestamp after reduction
                sub_chunk['__time'] = pd.to_datetime(sub_chunk['__time'], unit='s', errors='coerce')
                
                filtered_chunks.append(sub_chunk)
            
            print(f"Processed chunk {chunk_idx + 1}... Kept {len(sub_chunk)} rows matching criteria.")

if filtered_chunks:
    # Concatenate filtered parts
    final_df = pd.concat(filtered_chunks, ignore_index=True)
    
    print(f"\nFinal Filtered Dataset Shape: {final_df.shape}")

else:
    print("No rows matched the target metrics. Check your `desired_metrics` names against the peek output.")