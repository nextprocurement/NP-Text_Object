import argparse
from src.objective_tender_extraction.objective_extractor import ObjetiveExtractor
import pandas as pd


if __name__ == '__main__':
    
    # Parse arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--input', type=str, required=True, help='Path to the input file')
    #parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    
    oe = ObjetiveExtractor(do_train=False)
    
    df = pd.read_parquet('/export/data_ml4ds/NextProcurement/PLACE/pliegos_extracted/all_extracted_29jul_sample1000_es.parquet')
    
    try:
        df = oe.predict(df)
    except Exception as e:
        print(f'Error: {e}')

    import pdb; pdb.set_trace()
    
    