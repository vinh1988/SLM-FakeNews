import json
import os
import pandas as pd
from pathlib import Path

def extract_metrics_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract basic info
    result = {
        'dataset': data['dataset'],
        'model': data['model'],
        'num_train_samples': data['num_train_samples'],
        'num_test_samples': data['num_test_samples']
    }
    
    # Add metrics
    result.update(data['metrics'])
    
    # Add hyperparameters with 'hp_' prefix
    for k, v in data['hyperparameters'].items():
        result[f'hp_{k}'] = v
    
    return result

def main():
    results_dir = Path('/home/vinh/Documents/SLM-FakeNews/results_final')
    json_files = list(results_dir.glob('**/results.json'))
    
    all_metrics = []
    for json_file in json_files:
        try:
            metrics = extract_metrics_from_json(json_file)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Convert to DataFrame and save as CSV with proper formatting
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Select only the specified columns
        columns_to_keep = [
            'dataset', 'model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC',
            'samples_per_sec', 'Train_Time_s', 'Test_Loss', 'Test_Steps_per_sec'
        ]
        df = df[columns_to_keep]
        
        output_path = results_dir / 'all_metrics.csv'
        # Ensure consistent decimal formatting and clean output
        float_format = '%.6f'  # 6 decimal places for consistency
        df.to_csv(output_path, index=False, float_format=float_format, encoding='utf-8')
        
        # Verify the file was written correctly
        if output_path.exists():
            print(f"Successfully saved filtered metrics to {output_path}")
            print(f"Total rows: {len(df)} (including header)")
            print("\nFirst few rows:")
            print(df.head().to_string())
        else:
            print("Error: Failed to save the CSV file.")
    else:
        print("No metrics were extracted.")

if __name__ == "__main__":
    main()
