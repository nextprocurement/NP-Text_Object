import argparse
import pathlib
import subprocess

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the input file", default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_29jul_sample1000_es.parquet")
    parser.add_argument("--output", type=str, required=False, help="Path to the output file", default="/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives_preprocessed/all_extracted_29jul_sample1000_es.parquet")
    
    args = parser.parse_args()

    # Correct the replacement syntax for out_save
    out_save = pathlib.Path(str(args.output).replace("all_extracted_29jul_sample1000_es.parquet", "pliegos.parquet"))
    
    # Copy input file to temporary file
    subprocess.run(["cp", args.input, str(out_save)])

    preprocessing_script = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/src/preprocessing/pipe/nlpipe.py"
    source_path = str(out_save)
    source_type = "parquet"
    source = "pliegos"
    destination_path = args.output
    spacy_model = "es_core_news_md"
    lang = "es"

    # Construct the command
    cmd = [
        "python", preprocessing_script,
        "--source_path", source_path,
        "--source_type", source_type,
        "--source", source,
        "--destination_path", destination_path,
        "--lang", lang,
        "--spacy_model", spacy_model
    ]

    try:
        print(f'-- -- Running preprocessing command {" ".join(cmd)}')
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        print('-- -- Preprocessing failed. Revise command')
        print(e.output)
    print("-- -- Preprocessing done")
