import pandas as pd
from pathlib import Path
import zipfile

ROOT = 'High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare/'
NODE = ROOT + 'Node_Dataset/'
SINGLE = ROOT + 'Single_Machine_Dataset/'

target_csv = [
    # Node set (H100 / B200)
    NODE + 'Text Generation LLMs/B200/Batch_size/LLM_B200_8GPU_Node_Lamma8B_SeqLength2048_Deepspeedds_z3_BatchSize32.csv',
    NODE + 'Text Generation LLMs/H100/Batch_size/LLM_H100_8GPU_Node_Lamma8B_SeqLength2048_Deepspeedds_z3_BatchSize32.csv',
    NODE + 'Image Generation Diffusion Models/H100/Image_size_impact/Image_generation_H100_8GPU_Node_ModelSize128_ImageSize64_BatchSize128.csv',
    NODE + 'Image Generation Diffusion Models/B200/Image_size_impact/Image_generation_B200_8GPU_Node_ModelSize128_ImageSize128_BatchSize128.csv',
    # Single machine (RTX 3060)
    SINGLE + 'Text Generation/Sequance Length/Single_machine_SeqLen500_EmbedDim100_batchsize128.CSV',
    SINGLE + 'Feature Forecasting/Model Size/Single_machine_NumLayer6_SeqLength96_LayerSize1500_batchsize100.CSV',
]

# Only the columns we need
GPU_PREFIXES = (
    'GPU Temperature [', 'GPU Hot Spot Temperature [', 'GPU Core Voltage [',
    'GPU Fan1 [RPM]', 'GPU Power [W]', 'GPU Core Load [',
    'GPU Memory Controller Load [', 'GPU Memory Usage [', 'Total GPU Power [% of TDP]',
)

def node_cols(c):
    return c == 'timestamp' or c.startswith('gpu')

def single_cols(c):
    return c in ('Date', 'Time') or c.startswith(GPU_PREFIXES)

print("\nHigh-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare.zip:\n")
file_path1 = Path("data") / "High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare.zip"

with zipfile.ZipFile(file_path1, 'r') as z:
    for i in target_csv:
        try:
            z.extract(i, path='.')
        except KeyError:
            print(f"[WARNING] File not found in zip archive (skipping): {i}")
            continue
        print(f"Successfully extracted: {i}")

        try:
            is_node = '/Node_Dataset/' in i
            df = pd.read_csv(
                i,
                encoding='latin-1',
                usecols=node_cols if is_node else single_cols,
                low_memory=False,
            )

            if not is_node:
                # HWiNFO appends a repeated header row + the GPU name row at the end.
                # Drop them, then turn the sensor columns into numbers.
                df = df[df['Date'].str.match(r'^\d+\.\d+\.\d{4}$', na=False)].copy()
                sensor_cols = [c for c in df.columns if c not in ('Date', 'Time')]
                df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')

            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} cols")
            # node files: summarise GPU 0 only (all 8 GPUs have the same columns)
            view = df.filter(like='gpu0_') if is_node else df
            print(view.describe().T[['count', 'min', 'mean', 'max']])
            print("-" * 40)
        except Exception as e:
            print(f"[ERROR] Failed processing {i}: {e}")