import pandas as pd
import pathlib

path_lst = "/export/data_ml4ds/NextProcurement/temporal"
path_save = "/export/data_ml4ds/NextProcurement/temporal/pt_to_extract"


file_count = sum(1 for file in pathlib.Path(path_lst).rglob('*') if file.is_file())
for el, path in enumerate(pathlib.Path(path_lst).rglob('*')):
    if path.name.endswith("_correct.parquet"):
        print("*" * 50)
        print(f"-- -- Processing file {el+1} / {file_count}: {path}")
        print("*" * 50)
        
        df = pd.read_parquet(path)
        df = df[(df.json_path.str.contains("/ppt")) | (df.json_path.str.contains("/PPT"))]
        
        this_path_save = path_save + "/" + path.name
        
        df.to_parquet(this_path_save)

    
    