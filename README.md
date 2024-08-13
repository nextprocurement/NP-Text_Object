# NP-Text_Object

## Preprocessing Instructions

1. **Add NLPipe Repository**:
   Add the `NLPipe` repository as a submodule in the `src/preprocessing` directory:

   ```bash
   git submodule add https://github.com/Nemesis1303/NLPipe.git src/preprocessing/NLPipe

2. Stopwords Placement: To ensure effective removal of stopwords, place the stopwords file in the ``data/stw_lists/es`` directory (assuming the language is Spanish).
3. Create a separate environment based on the requirements specified in ``/src/preprocessing/requirements.txt`` to avoid conflicts with the libraries required by the extraction module.


## Important folders

(Located at ``/export/data_ml4ds/NextProcurement/PLACE``)

- **pliegos_extracted:** Documents extracted by BSC after XML removal and language detection. The final version of language detection is pending from BSC.
- **pliegos_objectives:** Results of applying objective_extractor.py on parquets from pliegos_extracted.
- **pliegos_objectives_preprocessed:** Results of applying nlp_preprocess.py on parquets from pliegos_objectives.