import logging
import pathlib
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
from typing import Optional, Union
import dspy
from dspy.datasets import Dataset
import pathlib
from sklearn.model_selection import train_test_split
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

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
        paths = [path for path in data_fpath.iterdir()]
        all_dfs = []
        for path_ in tqdm(paths):
            df = pd.read_excel(path_)
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
        desc="The tender objective, or 'N_A' if not present.")


class ObjetiveExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PredictObjective)

    def _process_output(self, text):

        if "N_A" in text:
            return "/"
        else:
            return text

    def forward(self, text):
        pred = self.predict(TENDER=text[0:5000])

        return dspy.Prediction(objective=self._process_output(pred.OBJECTIVE))


class ObjetiveExtractor(object):
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        path_tr_data="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/admin_eval_task/curated",
        trained_promt="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Text_Object/data/prompts/ObjetiveExtractor-saved.json",
        do_train=False,
        logger: logging.Logger = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent / "data/logs"
    ):

        self._logger = logger if logger else init_logger(__name__, path_logs)

        # Dspy settings
        if model_type == "llama":
            lm = dspy.HFClientTGI(model="meta-llama/Meta-Llama-3-8B ",
                                  port=8080, url="http://127.0.0.1")
        elif model_type == "openai":
            load_dotenv(path_open_api_key)
            api_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
            lm = dspy.OpenAI(model=open_ai_model)
        dspy.settings.configure(lm=lm)

        if not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("-- -- Trained prompt not found. Exiting.")
                return
            self.module = ObjetiveExtractorModule.load(trained_promt)

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

    def optimize_module(self, data_path, mbd=4, mld=16, ncp=16, mr=1, dev_size=0.25):

        # Create dataset
        dataset = TenderDataset(
            data_fpath=data_path,
            dev_size=dev_size,
        )

        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                      num_candidate_programs=ncp, max_rounds=mr)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.combined_score, **config)
        compiled_pred = teleprompter.compile(
            ObjetiveExtractorModule(), trainset=trainset, valset=devset)

        # Apply on test set
        tests = []
        for el in testset:
            output = compiled_pred(el.text)
            tests.append([el.text[0:5000], el.objetivo,
                         output["objective"], self.combined_score(el, output)])

        mean = pd.DataFrame(tests, columns=["TEXT", "GROUND", "PREDICTED", "METRIC"])[
            "METRIC"].mean()

        self._logger.info(f"-- -- Mean score on test set: {mean}")

        return compiled_pred
