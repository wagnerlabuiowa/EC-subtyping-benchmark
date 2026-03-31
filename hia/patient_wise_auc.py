import pandas as pd
import numpy as np
from sklearn import metrics
import os
import argparse
from eval.eval import CalculatePatientWiseAUC
import utils.utils as utils

def compute_patient_results(result_csv_path, target_label_dict, result_dir, report_file_path=None):
    """
    Compute patient-based results from tile-based results without re-running deployment.
    
    Args:
        result_csv_path: Path to TEST_RESULT_TILE_BASED_FULL.csv
        target_label_dict: Dictionary mapping class names to indices
        result_dir: Directory to save patient-based results
        report_file_path: Optional path to report file for logging
    """
    
    # Create a minimal args object
    class Args:
        def __init__(self):
            self.target_labelDict = target_label_dict
            self.result_dir = result_dir
    
    args = Args()
    
    # Create report file if needed
    if report_file_path:
        report_file = open(report_file_path, 'a', encoding="utf-8")
    else:
        report_file = open(os.path.join(result_dir, 'Patient_Results_Report.txt'), 'w', encoding="utf-8")
    
    try:
        # Call the existing function
        patient_result_path = CalculatePatientWiseAUC(
            resultCSVPath=result_csv_path,
            args=args,
            reportFile=report_file,
            foldcounter=None,
            clamMil=False
        )
        
        print(f"Patient-based results saved to: {patient_result_path}")
        return patient_result_path
        
    finally:
        report_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute patient-based results from tile-based results')
    parser.add_argument('--result_csv', type=str, required=True, 
                       help='Path to TEST_RESULT_TILE_BASED_FULL.csv')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Directory to save patient-based results')
    parser.add_argument('--report_file', type=str, default=None,
                       help='Optional path to report file')
    
    args = parser.parse_args()
    
    # Infer target labels from CSV columns
    df = pd.read_csv(args.result_csv, nrows=1)
    exclude_cols = {'PATIENT', 'TilePath', 'yTrue', 'yTrueLabel'}
    target_labels = [col for col in df.columns if col not in exclude_cols]
    target_label_dict = {label: i for i, label in enumerate(target_labels)}
    print(f"Inferred target labels: {target_labels}")
    
    compute_patient_results(
        result_csv_path=args.result_csv,
        target_label_dict=target_label_dict,
        result_dir=args.result_dir
    )