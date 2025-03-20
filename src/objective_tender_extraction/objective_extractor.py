import logging
import pathlib
import pickle
import pandas as pd
from tqdm import tqdm # type: ignore
import os
from dotenv import load_dotenv
from typing import Optional, Union
import dspy # type: ignore
from dspy.datasets import Dataset # type: ignore
import pathlib
from sklearn.model_selection import train_test_split # type: ignore
from dspy.teleprompter import BootstrapFewShotWithRandomSearch # type: ignore
from bert_score import score # type: ignore
from src.objective_tender_extraction.prompter import Prompter
from src.utils.utils import init_logger
from dspy.evaluate import Evaluate # type: ignore
import ujson
#from utils import find_object_pos, OBJETOS


#######################################################################
# TenderDataset
#######################################################################
class TenderDataset(Dataset):
    """Class to load the data in the format required by DSPy for training. It reads the data from a list of Excel files and splits it into training, development, and test sets. The Excel files are expected to have the following columns: 'procurement_id', 'doc_name', 'text', and 'objetivo'. It is assumed that these files have been generated via manual curation by public administrations.
    """

    def __init__(
        self,
        data_fpath: Union[pathlib.Path, str],
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        text_key: str = "text",
        seed: Optional[int] = 11235,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.labels = []
        self._train = []
        self._dev = []
        self._test = []

        # Read the training data
        paths = [path for path in pathlib.Path(data_fpath).iterdir()]
        all_dfs = []
        for path_ in tqdm(paths):
            df = pd.read_excel(path_)

            # Limit text to 4000 characters for training
            df["text"] = df["text"].apply(lambda x: x[0:4000])
            all_dfs.append(df)
        train_data = pd.concat(all_dfs)

        train_data, temp_data = train_test_split(
            train_data, test_size=dev_size + test_size, random_state=seed)
        dev_data, test_data = train_test_split(
            temp_data, test_size=test_size / (dev_size + test_size), random_state=seed)

        self._train = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(train_data)
        ]
        self._dev = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(dev_data)
        ]
        self._test = [
            dspy.Example({**row}).with_inputs(text_key) for row in self._convert_to_json(test_data)
        ]

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')

#######################################################################
# ExtractObjective
#######################################################################
class PredictObjective(dspy.Signature):
    """
    Extract the objective of the contract from a document containing the technical specifications of a Spanish public tender. If the objective is not present in the document, return '/'.

    Requirements:

    The extracted text must exclusively consist of words from the document. No additional words are allowed.
    The language of the document must remain unchanged under all circumstances.
    """

    TENDER = dspy.InputField(
        desc="The document containing the technical specifications of the Spanish public tender.")
    OBJECTIVE = dspy.OutputField(
        desc="The tender objective, or '/' if not present.")

class ObjetiveExtractorModule(dspy.Module):
    """
    DSPy module to extract the objective of a Spanish public tender from a document containing the technical specifications. If the objective is not present in the document, return '/'.
    """
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PredictObjective)

        self.no_objective_variations = [
            '/', '/ (no present)', '/ (no objective present in the document)',
            'No objective is present in the document.', '/ (not present)',
            '/ (No objective present in the document)',
            '/ (not present in the document)', "'/'", 'N/A', '/.',
            '/ (no objective present in the document).',
            '/ (No objective present in the document).',
            'No objective present.', "'/' (not present in the document)",
            '/ (No present)', "'/'.", 'No present.',
            '/ (no objective is present in the document)',
            'No objective is present in this document.',
            'The objective of the tender is not present in the document.'
        ]

    def _process_output(self, text):
        """
        Process the output of the model to ensure that the objective is not one of the variations of 'no objective present in the document'.
        """

        if text in self.no_objective_variations or "N_A" in text or "N/A" in text or "NA" in text:
            return "/"
        else:
            return text

    def forward(self, text):
        """Forward method to extract the objective from the document."""
        pred = self.predict(TENDER=text)

        return dspy.Prediction(objective=self._process_output(pred.OBJECTIVE))

#######################################################################
# ObjetiveExtractor
#######################################################################
class ObjetiveExtractor(object):
    """Wrapper class for the ObjetiveExtractorModule. It can be used to load a pre-trained module or to train a new one. The training data is expected to be in the format of a list of Excel files, each containing the columns 'procurement_id', 'doc_name', 'text', and 'objetivo'. The training process uses the BootstrapFewShotWithRandomSearch teleprompter to optimize the module. The trained module can be saved to a file and loaded later for predictions.
    """
    def __init__(
        self,
        model_type: str = "ollama_chat/qwen2.5:32b", #"ollama_chat/llama3:70b-instruct",
        ollama_url: str = "http://127.0.0.1:11434" ,
        open_ai_model: str = None,
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        path_tr_data="data/admin_eval_task/curated",
        trained_promt="data/prompts/ObjetiveExtractor-saved.json",
        do_train=False,
        logger: logging.Logger = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs"
    ):
        """
        Initialize the ObjetiveExtractor object.
        
        Parameters
        ----------
        model_type : str, optional
            The model type to use for the ObjetiveExtractorModule, by default "ollama_chat/qwen2.5:32b".
        ollama_url : str, optional
            The URL of the Ollama server, by default the default ollama url and port.
        open_ai_model : str, optional
            The OpenAI model to use, by default None.
        path_open_api_key : str, optional
            The path to the OpenAI API key.
        path_tr_data : str, optional
            The path to the training data, by default "data/admin_eval_task/curated"
        trained_promt : str, optional
            The path to save the trained module, by default "data/prompts/ObjetiveExtractor-saved.json"
        do_train : bool, optional
            Whether to train the module, by default False
        logger : logging.Logger, optional
            The logger to use, by default None
        path_logs : pathlib.Path, optional
            The path to save the logs, by default pathlib.Path(__file__).parent.parent.parent / "data/logs"    
        """
        self._logger = logger if logger else init_logger(__name__, path_logs)

        # Dspy settings            
        if open_ai_model:
            load_dotenv(path_open_api_key)
            api_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
            self.lm = dspy.OpenAI(model=open_ai_model)
            
        else:
            self.lm = dspy.LM(
                model_type=model_type,
                api_base=ollama_url 
            )
        dspy.settings.configure(lm=self.lm)
        

        if not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("-- -- Trained prompt not found. Exiting.")
                return
            oe = ObjetiveExtractorModule()
            oe.load(trained_promt)
            self.module = oe            
            
            """Only if prompter is used  
            self.prompter =  Prompter(
                model_type="qwen2.5:32b")  
            INSTRUCTIONS_PATH = "/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/src/objective_tender_extraction/templates/extract_objective_v1.txt"
            with open(INSTRUCTIONS_PATH, 'r') as file: self.template = file.read()
            """

            self._logger.info(
                f"-- -- ObjetiveExtractorModule loaded from {trained_promt}")
        else:
            if not path_tr_data:
                self._logger.error(
                    "-- -- Data path is required for training. Exiting.")
                return
            else:
                self._logger.info(
                    f"-- -- Training ObjetiveExtractorModule from {path_tr_data}")
                self.module = self.optimize_module(path_tr_data)
                self.module.save(trained_promt)
                self._logger.info(
                    f"-- -- ObjetiveExtractorModule trained and saved to {trained_promt}")

    def combined_score(self, example, pred, trace=None):
        """
        Calculate the combined score of the prediction based on the Jaccard similarity and the presence of the predicted objective in the text.
        
        Parameters
        ----------
        example : dspy.Example
            The example containing the text and the objective.
        pred : dict
            The prediction containing the objective.
        trace : dict, optional
            The trace of the prediction, by default None
                
        Returns
        -------
        float
            The combined score of the prediction, calculated as:
            0.5 * Jaccard similarity + 0.5 * Presence of the predicted objective in the text.
        """
        
        def matching_score(example, pred, trace=None):
            if example.objetivo == "/":
                if pred["objective"] == "/":
                    return 1.0
                else:
                    return 0.0

            predicted_lst = pred["objective"].split()
            gt_lst = example.objetivo.split()

            predicted_set = set(predicted_lst)
            gt_set = set(gt_lst)

            intersection = predicted_set.intersection(gt_set)
            union = predicted_set.union(gt_set)

            if len(union) == 0:
                return 0.0
            jaccard_similarity = len(intersection) / len(union)

            return jaccard_similarity

        def is_in_text_score(example, pred, trace=None):
            if example.objetivo == "/":
                if pred["objective"] == "/":
                    return 1.0
                else:
                    return 0.0

            text_lst = example.text[0:5000].lower().split()
            predicted_lst = pred["objective"].lower().split()

            words_not_in_text = [
                word for word in predicted_lst if word not in text_lst]
            num_words_not_in_text = len(words_not_in_text)

            total_predicted_words = len(predicted_lst)
            score = max(
                0.0, 1.0 - (num_words_not_in_text / total_predicted_words))

            return score

        match_score = matching_score(example, pred, trace)
        text_score = is_in_text_score(example, pred, trace)
        combined = (0.5 * match_score) + (0.5 * text_score)

        return combined

    def get_bert_score(self, df):
        """Calculate the BERT score of the predictions on the ground truth.
        """

        model_name = "microsoft/deberta-xlarge-mnli"

        P, R, F1 = score(df.PREDICTED.values.tolist(
        ), df.GROUND.values.tolist(), lang='es', model_type=model_name)

        df["P"] = P
        df["R"] = R
        df["F1"] = F1

        return df

    def get_in_text_score(self, df, col_extract, objective_column, start_token, nr_tokens): 
        """Calculate the score based on the presence of the predicted objective in the text.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the text and the predicted objective.
        col_extract : str
            The column containing the text.
        objective_column : str
            The column containing the predicted objective.
        start_token : int
            The start token of the text.
        nr_tokens : int
            The number of tokens to consider.
            
        Returns
        -------
        float
            The score based on the presence of the predicted objective in the text.
        """
        if df[objective_column] == "/":
            return 0.0

        text_lst = df[col_extract][start_token:start_token + nr_tokens].lower().split()
        predicted_lst = df[objective_column].lower().split()

        words_not_in_text = [
            word for word in predicted_lst if word not in text_lst]
        num_words_not_in_text = len(words_not_in_text)

        total_predicted_words = len(predicted_lst)
        
        second = 0.0
        try:
            second = 1.0 - (num_words_not_in_text / total_predicted_words)
        except ZeroDivisionError:
            self._logger.error("-- -- ZeroDivisionError in get_in_text_score")
        score = max(0.0, second)

        return score

    def optimize_module(self, data_path, mbd=16, mld=64, ncp=64, mr=5, dev_size=0.25):
        """Optimize the ObjetiveExtractorModule using the BootstrapFewShotWithRandomSearch teleprompter.
        """

        # Create dataset
        dataset = TenderDataset(
            data_fpath=data_path,
            dev_size=dev_size,
        )

        self._logger.info(f"-- -- Dataset loaded from {data_path}")

        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        self._logger.info(
            f"-- -- Dataset split into train, dev, and test. Training module...")

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,num_candidate_programs=ncp, max_rounds=mr)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.combined_score, **config)

        compiled_pred = teleprompter.compile(
            ObjetiveExtractorModule(), trainset=trainset, valset=devset)

        self._logger.info(f"-- -- Module compiled. Evaluating on test set...")

        # Apply on test set
        tests = []
        for el in testset:
            output = compiled_pred(el.text[0:5000])
            tests.append([el.text[0:5000], el.objetivo, output["objective"], self.combined_score(el, output)])

        df = pd.DataFrame(
            tests, columns=["TEXT", "GROUND", "PREDICTED", "METRIC"])
        df = self.get_bert_score(df)

        self._logger.info(
            f"-- -- BERT score on test set: {df[['P', 'R', 'F1']].mean()}")
        self._logger.info(
            f"-- -- Mean score on test set: {df['METRIC'].mean()}")
        print(
            f"-- -- BERT score on test set: {df[['P', 'R', 'F1']].mean()}")
        print(
            f"-- -- Mean score on test set: {df['METRIC'].mean()}")

        evaluate = Evaluate(
            devset=devset, metric=self.combined_score, num_threads=1, display_progress=True)
        compiled_score = evaluate(compiled_pred)
        uncompiled_score = evaluate(ObjetiveExtractorModule())

        print(
            f"## ObjetiveExtractorModule Score for uncompiled: {uncompiled_score}")
        print(
            f"## ObjetiveExtractorModule Score for compiled: {compiled_score}")
        print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")

        return compiled_pred

    def predict(self, df, col_extract, token_starts=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], checkpoint_path="checkpoint.pkl", save_interval=10):
        """Predict the objective of the contract from a document containing the technical specifications of a Spanish public tender. If the objective is not present in the document, return '/'.
        """
        
        def extract_objective_and_score(df, col_extract, start_token=0, score_column="in_text_score", objective_column="objective"):
            
            processed_column = f'processed_{start_token}'
            for index, row in tqdm(df.iterrows(), total=len(df)):
                print(f"Index {index}")
                if row[processed_column]:
                    continue  # Skip rows that are already processed for this token_start
                
                nr_tokens = 3000
                while True:
                    try:
                        # ---------------------------------------------------
                        # USING MODULE
                        # ---------------------------------------------------
                        extracted_text = row[col_extract][start_token:start_token + nr_tokens]       
                        print(f"Extracted text for index {index} (first 100 chars): {extracted_text[:100]}")
                        objective = self.module(extracted_text)["objective"]
                        # ---------------------------------------------------
                        
                        # ---------------------------------------------------
                        # USING PROMPTER
                        # ---------------------------------------------------
                        """
                        word, pos = find_object_pos(row[col_extract], OBJETOS)
                        print(f"Word: {word} --> Pos: {pos}")
                        if word is None:
                            extracted_text = row[col_extract][start_token:start_token + nr_tokens]
                        else:
                            extracted_text = row[col_extract][pos:pos + nr_tokens]
                            old_token_start = start_token
                            start_token = pos
                        print(
                            f"Extracted text for index {index} (first 100 chars): {extracted_text[:200]}")
                        
                        template = self.template.format(document=extracted_text)
                        objective, _ = self.prompter.prompt(template)
                        
                        if objective == "/":
                            print(f"Objective not found in the text. Trying with full text")
                            extracted_text = row[col_extract][old_token_start:old_token_start + 3000]
                            template = self.template.format(document=extracted_text)
                            objective, _ = self.prompter.prompt(template)
                        """
                        # ---------------------------------------------------
                        
                        # print extracted objective in a different color
                        print(f"\033[92mTOKEN START: {start_token} --> OBJECTIVE: \n {objective}\033[0m")
    
                        # save objective in the dataframe
                        df.loc[index, objective_column] = objective

                        print(
                            f"DataFrame updated with {objective_column} for index {index}")
                        break
                    except Exception as e:
                        print(f"Exception at index {index}: {e}")
                        nr_tokens -= 500
                        if nr_tokens <= 0:
                            df.loc[index, objective_column] = None
                            print(
                                f"{objective_column} set to None for index {index}")
                            break
                score = self.get_in_text_score(df.loc[index], col_extract, objective_column, start_token, nr_tokens)
                print(f"Score for index {index}: {score}")
                df.loc[index, score_column] = score
                
                # Mark this row as processed for this token_start
                df.loc[index, processed_column] = True

                # Save checkpoint after every `save_interval` rows
                if index % save_interval == 0:
                    print(f"-- -- Saving checkpoint to {checkpoint_path} at row {index}")
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(df, f)

            return df

        def process_extractions(df, col_extract, score_column, objective_column):
            best_scores = {}
            best_objectives = {}
            replacement_logs = []

            for start_token in token_starts:
                print("*" * 50)
                print(f"-- -- Processing start token {start_token}")
                print("*" * 50)
                processed_column = f'processed_{start_token}'
                if processed_column not in df.columns:
                    df[processed_column] = False  # Initialize processed column for this token_start

                df_temp = extract_objective_and_score(df.copy(
                ), col_extract, start_token=start_token, score_column=score_column, objective_column=objective_column)
                replacements = 0

                for index, row in df_temp.iterrows():
                    current_score = row[score_column]
                    if index not in best_scores or current_score > best_scores[index]:
                        if index in best_scores:
                            replacements += 1
                        best_scores[index] = current_score
                        best_objectives[index] = row[objective_column]

                df[processed_column] = df_temp[processed_column]
                
                replacement_logs.append((start_token, replacements))
                print(
                    f"Replacements in iteration with start token {start_token}: {replacements}")

            for index in df.index:
                df.loc[index, objective_column] = best_objectives.get(
                    index, None)
                df.loc[index, score_column] = best_scores.get(index, 0.0)

            return df, replacement_logs

        # Load checkpoint if it exists
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                df = pickle.load(f)

        # Initialize columns if not already done
        if "objective" not in df.columns:
            df["objective"] = None
        if "in_text_score" not in df.columns:
            df["in_text_score"] = None

        # Perform extractions and get the best results
        df, replacement_logs = process_extractions(df, col_extract, score_column="in_text_score", objective_column="objective")

        # Final save at the end of processing
        print(f"Final save to {checkpoint_path}")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(df, f)

        # Print the replacement logs
        print("Summary of replacements in each iteration:")
        for start_token, replacements in replacement_logs:
            print(f"Start token {start_token}: {replacements} replacements")

        return df