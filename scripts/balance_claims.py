import argparse
import pandas as pd
from sklearn.utils import resample
import yaml
from pathlib import Path


def load_config(path='config/config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def balance_by_upsampling(df, target='denial_flag', factor=2.0):
    '''
        Randomly upsample the minority class by a factor (e.g., 2.0 = double it).
    '''
    majority_class = df[df[target] == 1]
    minority_class = df[df[target] == 0]

    upsampled = resample(
        minority_class,
        replace=True,
        n_samples=int(len(minority_class) * factor),
        random_state=42
    )

    df_balanced = pd.concat([majority_class, upsampled])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='Path to input CSV')
    parser.add_argument('--output', required=True, help='Path to save balanced CSV')
    parser.add_argument('--factor', type=float, default=None, help='Upsample the minority class by a factor')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    # Load config and determine input path
    config = load_config(args.config)
    balanced_path = Path(config['data']['balanced_path'])
    default_path = Path(config['data']['default_path'])

    # Use balanced if it exists, else fallback to default
    if balanced_path.exists():
        data_path = balanced_path
    else:
        data_path = default_path
    
    input_path = args.input if args.input else data_path

    # Load data
    df = pd.read_csv(input_path)

    # Use default factor if not provided
    factor = args.factor if args.factor is not None else 2.0

    # Balance data
    df_balanced = balance_by_upsampling(df, factor=factor)

    # Save result
    df_balanced.to_csv(args.output, index=False)

    # Display result
    print("Balanced class distribution:")
    print(df_balanced['denial_flag'].value_counts(normalize=True).rename("proportion"))
