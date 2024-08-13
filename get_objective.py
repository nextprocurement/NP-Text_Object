import argparse
from src.objective_tender_extraction.objective_extractor import ObjetiveExtractor
import pandas as pd
import time

if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False, help='Path to the input file', default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_extracted/all_extracted_12aug_es.parquet")
    parser.add_argument('--output', type=str, required=False, help='Path to the output file', default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_12aug_es.parquet")
    
    args = parser.parse_args()
    
    oe = ObjetiveExtractor(do_train=False)
    
    df = pd.read_parquet(args.input)
        
    time_start = time.time()
    try:
        print("-- -- Extracting objectives")
        df = oe.predict(df)
    except Exception as e:
        print(f'Error: {e}')
    
    time_end = time.time()
    
    print(
        f'-- -- Time elapsed for objective extraction: {time_end - time_start}')
    print(f'-- -- Saving extracted objectives to {args.output}')
    df.to_parquet(args.output)

    print("-- -- STATS:")
    len_no_found = len(df[df.objective == "/"])
    avg_score = df[df.objective != "/"].in_text_score.mean()
    print(f'-- -- -- Number of no found objectives: {len_no_found}')
    print(f'-- -- -- Average score: {avg_score}')
    
    
    