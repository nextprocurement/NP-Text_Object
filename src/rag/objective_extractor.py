import argparse
import heapq
import logging
import pathlib

from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore

from llama_index.core import VectorStoreIndex, Document  # type: ignore
from llama_index.core.node_parser import SentenceSplitter  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore
from prompter import Prompter

from file_utils import load_yaml_config_file, init_logger


class ObjectiveExtractor(object):
    def __init__(
        self,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("src/rag/config/config.yaml"),
        **kwargs
    ):
        
        self._logger = logger if logger else init_logger(config_path, __name__)
        config = load_yaml_config_file(config_path, "extractor", logger)

        # Merge config with any additional keyword arguments
        config = {**config, **kwargs}
        
        self.embed_model = HuggingFaceEmbedding(model_name=config.get("embedding_model"))
        self.node_parser = SentenceSplitter(chunk_size=config.get("chunk_size"), chunk_overlap=config.get("chunk_overlap"))
        self._logger.info(f"Initializing prompter with model type: {config.get('llm_model_type')}")
        self.prompter = Prompter(model_type=config.get("llm_model_type"))
        self.calculate_on = config.get("calculate_on")
        
        with open(config.get("templates", {}).get("generative", "")) as f:
            self.generative_prompt = f.read()

        with open(config.get("templates", {}).get("extractive", "")) as f:
            self.extractive_prompt = f.read()
            
        self._logger.info("ObjectiveExtractor initialized with config: %s", config_path)
        
    def extract(self, text, option="generative"):
        try:
            doc = Document(text=text)
            nodes = self.node_parser.get_nodes_from_documents([doc])

            # Setup retrievers
            vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model)
            vector_retriever = vector_index.as_retriever(similarity_top_k=4)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)

            # Combine results manually
            query = "objeto del contrato, objeto de la contratación, tiene por objeto, objetivos del contrato, objeto del pliego, objectivo"
            retrieved_nodes = self._combine_retrievers([bm25_retriever, vector_retriever], query)

            # Create prompt and run
            context = "\n\n".join([n.get_content() for n in retrieved_nodes])
            if option == "generative":
                prompt = self.generative_prompt.format(context=context)
            elif option == "extractive":
                prompt = self.extractive_prompt.format(context=context)
            else:
                raise ValueError("Invalid option. Use 'generative' or 'extractive'.")
            #import pdb; pdb.set_trace()
            result, _ = self.prompter.prompt(question=prompt, use_context=False)
            return result.strip()
        except Exception as e:
            return f"ERROR: {e}"

    def _combine_retrievers(self, retrievers, query, top_k=4):
        all_nodes = []
        for retriever in retrievers:
            all_nodes.extend(retriever.retrieve(query))

        unique_nodes = {}
        for n in all_nodes:
            nid = n.node.node_id
            if nid not in unique_nodes or (n.score or 0) > (unique_nodes[nid].score or 0):
                unique_nodes[nid] = n

        return heapq.nlargest(top_k, unique_nodes.values(), key=lambda x: x.score or 0)

    def apply_to_dataframe(self, df):
        tqdm.pandas()
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")

        if self.calculate_on not in df.columns:
            raise ValueError(f"Column '{self.calculate_on}' not found in DataFrame.")

        # Extractive objective extraction
        time_start = pd.Timestamp.now()
        self._logger.info(f"Applying extractive objective extraction to column '{self.calculate_on}'")
        df["extracted_objective"] = df[self.calculate_on].progress_apply(
            lambda text: self.extract(text, option="extractive")
        )
        time_end = pd.Timestamp.now()
        self._logger.info("Extractive objective extraction completed in %.2f seconds", (time_end - time_start).total_seconds())

        # Generative objective extraction
        time_start = pd.Timestamp.now()
        self._logger.info(f"Applying generative objective extraction to column '{self.calculate_on}'")
        df["generated_objective"] = df[self.calculate_on].progress_apply(
            lambda text: self.extract(text, option="generative")
        )
        time_end = pd.Timestamp.now()
        self._logger.info("Generative objective extraction completed in %.2f seconds", (time_end - time_start).total_seconds())

        return df

        
def main():
    arparser = argparse.ArgumentParser(description="Objective Extractor")
    arparser.add_argument("--config", type=str, default="src/rag/config/config.yaml", help="Path to the configuration file")
    arparser.add_argument("--path_to_parquet", type=str, default="/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_chunks/part_0000.parquet", help="Path to the input parquet file")
    arparser.add_argument("--path_save", type=str, default="/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace_withExtracted", help="Path to save the output parquet file")
    arparser.add_argument("--calculate_on", type=str, default="texto_tecnico", help="Column to calculate the objective on")
    arparser.add_argument("--llm_model_type", type=str, default="llama3.1:8b", help="LLM model type to use for extraction")
    
    args = arparser.parse_args()
    
    extractor = ObjectiveExtractor(
        config_path=pathlib.Path(args.config),
        calculate_on=args.calculate_on,
        llm_model_type=args.llm_model_type
    )
    
    # read parquet file
    df = pd.read_parquet(args.path_to_parquet)
    if args.calculate_on == "texto_administrativo":
        df = df[df.resultado_administrativo == "Descargado correctamente"]
    elif args.calculate_on == "texto_tecnico":
        df = df[df.resultado_tecnico == "Descargado correctamente"]
    extractor._logger.info("Loaded dataframe with %d rows", len(df))
    
    # @TODO: remove this  
    df = df.sample(n=20, random_state=42)
    
    # enusre path save exists
    extractor._logger.info(f"Creating save path: {args.path_save}")
    path_save = pathlib.Path(args.path_save)
    path_save = path_save / pathlib.Path(args.path_to_parquet).name
    path_save.parent.mkdir(parents=True, exist_ok=True)
    
    extractor._logger.info(f"Extracting objectives from {len(df)} rows in column '{args.calculate_on}'")
    df = extractor.apply_to_dataframe(df)
    
    # Save the dataframe to parquet
    extractor._logger.info("Saving dataframe to %s", path_save)
    df.to_parquet(path_save, index=False)
    extractor._logger.info("Dataframe saved to %s", path_save)

if __name__ == "__main__":
    main()