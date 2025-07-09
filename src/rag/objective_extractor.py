import argparse
import heapq
import logging
import pathlib
from langdetect import detect # type: ignore
import ctranslate2 # type: ignore
import pyonmttok # type: ignore
from huggingface_hub import snapshot_download # type: ignore

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
        ollama_host: str = "http://kumo01.tsc.uc3m.es:11434",
        **kwargs
    ):
        
        self._logger = logger if logger else init_logger(config_path, __name__)
        config = load_yaml_config_file(config_path, "extractor", logger)

        # Merge config with any additional keyword arguments
        config = {**config, **kwargs}
        
        self.embed_model = HuggingFaceEmbedding(model_name=config.get("embedding_model"))
        
        model_dir = snapshot_download(repo_id=config.get("translation_model"), revision="main")
        self.ct2_tokenizer = pyonmttok.Tokenizer(mode="none", sp_model_path=f"{model_dir}/spm.model")
        self.ct2_translator = ctranslate2.Translator(model_dir)                
    
        self.node_parser = SentenceSplitter(chunk_size=config.get("chunk_size"), chunk_overlap=config.get("chunk_overlap"))
        self._logger.info(f"Initializing prompter with model type: {config.get('llm_model_type')}")
        self.prompter = Prompter(model_type=config.get("llm_model_type"), ollama_host=ollama_host)
        self.calculate_on = config.get("calculate_on")
        self.top_k = config.get("top_k")
        
        with open(config.get("templates", {}).get("generative", "")) as f:
            self.generative_prompt = f.read()

        with open(config.get("templates", {}).get("extractive", "")) as f:
            self.extractive_prompt = f.read()
            
        self._logger.info("ObjectiveExtractor initialized with config: %s", config_path)
        
    def translate_ca_to_es(self, text: str) -> str:
        tokenized = self.ct2_tokenizer.tokenize(text)
        translated = self.ct2_translator.translate_batch([tokenized[0]])
        return self.ct2_tokenizer.detokenize(translated[0][0]['tokens'])
        
    def extract(self, text, option="generative"):
        try:
            doc = Document(text=text)
            nodes = self.node_parser.get_nodes_from_documents([doc])
            
            top_k = min(self.top_k, len(nodes))

            # Setup retrievers
            vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model)
            vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

            # Combine results manually
            query = "objeto del contrato, objeto de la contratación, tiene por objeto, objetivos del contrato, objeto del pliego, objectivo"
            retrieved_nodes = self._combine_retrievers([bm25_retriever, vector_retriever], query, top_k=top_k)

            # Create prompt and run
            context = [n.get_content() for n in retrieved_nodes]
            
            # detect language in each context fragment
            detected_languages = [detect(fragment) for fragment in context]
            
            # if 75 % (len(context)) has catalan as language, then translate to spanish using https://huggingface.co/projecte-aina/aina-translator-es-ca
            catalan_count = detected_languages.count('ca')
            catalan_ratio = catalan_count / len(context)
            if catalan_ratio >= 0.75:
                self._logger.info(f"Detected {catalan_count} Catalan fragments out of {len(context)}. Translating to Spanish.")
                # Translate each Catalan fragment
                context = [
                    self.translate_ca_to_es(fragment) if lang == 'ca' else fragment
                    for fragment, lang in zip(context, detected_languages)
                ]
                #print(f"translated context: {context}")
            context_joint = "\n\n".join([c for c in context])
            if option == "generative":
                prompt = self.generative_prompt.format(context=context_joint)
            elif option == "extractive":
                prompt = self.extractive_prompt.format(context=context_joint)
            else:
                raise ValueError("Invalid option. Use 'generative' or 'extractive'.")
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

    def apply_to_dataframe(self, df, mode="both"):
        tqdm.pandas()
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")

        if self.calculate_on not in df.columns:
            raise ValueError(f"Column '{self.calculate_on}' not found in DataFrame.")

        valid_modes = {"extractive", "generative", "both"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")

        if mode in ("extractive", "both"):
            time_start = pd.Timestamp.now()
            self._logger.info(f"Applying extractive objective extraction to column '{self.calculate_on}'")
            df["extracted_objective"] = df[self.calculate_on].progress_apply(
                lambda text: self.extract(text, option="extractive")
            )
            time_end = pd.Timestamp.now()
            self._logger.info("Extractive objective extraction completed in %.2f seconds", (time_end - time_start).total_seconds())

        if mode in ("generative", "both"):
            time_start = pd.Timestamp.now()
            self._logger.info(f"Applying generative objective extraction to column '{self.calculate_on}'")
            df["generated_objective"] = df[self.calculate_on].progress_apply(
                lambda text: self.extract(text, option="generative")
            )
            time_end = pd.Timestamp.now()
            self._logger.info("Generative objective extraction completed in %.2f seconds", (time_end - time_start).total_seconds())

        return df
        
def main():
    argparser = argparse.ArgumentParser(description="Objective Extractor")
    argparser.add_argument("--config", type=str, default="src/rag/config/config.yaml", help="Path to the configuration file")
    argparser.add_argument("--path_to_parquet", type=str, default="/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_chunks/part_0004.parquet", help="Path to the input parquet file")
    argparser.add_argument("--path_save", type=str, default="/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace_withExtracted", help="Path to save the output parquet file")
    argparser.add_argument("--calculate_on", type=str, default="texto_tecnico", help="Column to calculate the objective on")
    argparser.add_argument("--llm_model_type", type=str, default="llama3.1:8b", help="LLM model type to use for extraction")
    argparser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model to use for vectorization")
    argparser.add_argument("--top_k", type=int, default=4, help="Number of top results to retrieve from the vector store")
    argparser.add_argument("--mode_extractive_generative", type=str, default="both", choices=["extractive", "generative", "both"], help="Mode of extraction: 'extractive', 'generative', or 'both'") 
    argparser.add_argument("--ollama_host", type=str, default="http://kumo01.tsc.uc3m.es:11434", help="Ollama host URL for LLM requests")
    
    args = argparser.parse_args()
    
    extractor = ObjectiveExtractor(
        config_path=pathlib.Path(args.config),
        ollama_host=args.ollama_host,
        calculate_on=args.calculate_on,
        llm_model_type=args.llm_model_type,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )
    
    # read parquet file
    df = pd.read_parquet(args.path_to_parquet)
    if args.calculate_on == "texto_administrativo":
        df = df[df.resultado_administrativo == "Descargado correctamente"]
    elif args.calculate_on == "texto_tecnico":
        df = df[df.resultado_tecnico == "Descargado correctamente"]
    extractor._logger.info("Loaded dataframe with %d rows", len(df))
    
    # @TODO: remove this  
    #df = df.sample(n=20, random_state=42)
    
    # enusre path save exists
    extractor._logger.info(f"Creating save path: {args.path_save}")
    path_save = pathlib.Path(args.path_save)
    path_save = path_save / pathlib.Path(args.path_to_parquet).name
    path_save.parent.mkdir(parents=True, exist_ok=True)
    
    extractor._logger.info(f"Extracting objectives from {len(df)} rows in column '{args.calculate_on}'")
    df = extractor.apply_to_dataframe(df, mode=args.mode_extractive_generative)
    
    # Save the dataframe to parquet
    extractor._logger.info("Saving dataframe to %s", path_save)
    df.to_parquet(path_save, index=False)
    extractor._logger.info("Dataframe saved to %s", path_save)

if __name__ == "__main__":
    main()