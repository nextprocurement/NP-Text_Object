import argparse
import pathlib
import subprocess
from tqdm import tqdm

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the input file", default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/to_process")
    #parser.add_argument("--input", type=str, required=False, help="Path to the input file", default="/export/usuarios_ml4ds/lbartolome/NextProcurement/sproc/place_feb_21/preprocessed")
    parser.add_argument("--output", type=str, required=False, help="Path to the output file", default="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/train_data/to_process")
    #parser.add_argument("--output", type=str, required=False, help="Path to the output file", default="/export/usuarios_ml4ds/lbartolome/NextProcurement/sproc/place_feb_21/preprocessed")
    
    args = parser.parse_args()
    print(f"Input path: {args.input}")
    for el in tqdm(pathlib.Path(args.input).rglob('*')):
        print(f"Processing {el}")
        this_out_save = pathlib.Path(args.output).joinpath(f"{el.stem}.parquet")

        # Correct the replacement syntax for out_save
        out_save = this_out_save.as_posix().replace(f"{el.stem}.parquet", "pliegos.parquet")
        
        if not this_out_save.exists():
            # Copy input file to temporary file
            subprocess.run(["cp", el.as_posix(), str(out_save)])
        else:
            print(f"Not recreating {this_out_save} since it already exists")

        preprocessing_script = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/src/preprocessing/pipe/nlpipe.py"
        source_path = str(out_save)
        source_type = "parquet"
        source = "pliegos"
        destination_path = this_out_save.as_posix()
        spacy_model = "es_dep_news_trf"
        lang = "es"
        embeddings_model = "paraphrase-multilingual-MiniLM-L12-v2"

        # Construct the command
        cmd = [
            "python", preprocessing_script,
            "--source_path", source_path,
            "--source_type", source_type,
            "--source", source,
            "--destination_path", destination_path,
            "--lang", lang,
            "--spacy_model", spacy_model,
            "--embeddings_model", embeddings_model,
            "--do_embeddings",
            "--no_preproc"
        ]

        try:
            print(f'-- -- Running preprocessing command {" ".join(cmd)}')
            subprocess.check_output(cmd)
        except subprocess.CalledProcessError as e:
            print('-- -- Preprocessing failed. Revise command')
            print(e.output)
        print("-- -- Preprocessing done")
