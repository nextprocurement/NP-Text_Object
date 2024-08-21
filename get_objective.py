import argparse
import pathlib
from src.objective_tender_extraction.objective_extractor import ObjetiveExtractor
import pandas as pd
import time

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False, help='Path to the input file', default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_extracted/all_extracted_12aug_es_parts")
    # parser.add_argument('--input', type=str, required=False, help='Path to the input file', default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_extracted/all_extracted_12aug_es.parquet")
    parser.add_argument('--output', type=str, required=False, help='Path to the output file', default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_12aug_es")
    # parser.add_argument('--output', type=str, required=False, help='Path to the output file', default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_12aug_es.parquet")
    parser.add_argument('--path_checkpoints', type=str, required=False, help='Path to the checkpoint files', default="/export/data_ml4ds/NextProcurement/PLACE/checkpoints")

    args = parser.parse_args()

    oe = ObjetiveExtractor(do_train=False)

    file_count = sum(1 for file in pathlib.Path(args.input).rglob('*') if file.is_file())
    for el, path in enumerate(pathlib.Path(args.input).rglob('*')):
        print("*" * 50)
        print(f"-- -- Processing file {el+1} / {file_count}: {path}")
        print("*" * 50)
        path_checkpoint = pathlib.Path(args.path_checkpoints).joinpath(
            f"{path.stem}_objectives.pkl")
        path_save = pathlib.Path(args.output).joinpath(
            f"{path.stem}_objectives.parquet")

        df = pd.read_parquet(path)

        time_start = time.time()
        try:
            print("-- -- Extracting objectives")
            df = oe.predict(df, checkpoint_path=path_checkpoint.as_posix())
        except Exception as e:
            print(f'Error: {e}')

        time_end = time.time()

        print(
            f'-- -- Time elapsed for objective extraction: {time_end - time_start}')

        print(f'-- -- Saving extracted objectives to {path_save.as_posix()}')

        df.to_parquet(path_save)

        print("-- -- STATS:")
        len_no_found = len(df[df.objective == "/"])
        avg_score = df[df.objective != "/"].in_text_score.mean()
        print(f'-- -- -- Number of no found objectives: {len_no_found}')
        print(f'-- -- -- Average score: {avg_score}')
