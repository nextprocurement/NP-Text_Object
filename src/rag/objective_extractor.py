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
        
        self._logger.info(f"Initializing prompter EXTRACTIVE with model type: {config.get('llm_model_type_ex')}")
        self.prompter_ex = Prompter(model_type=config.get("llm_model_type_ex"), ollama_host=ollama_host)
        
        self._logger.info(f"Initializing prompter GENERATIVE with model type: {config.get('llm_model_type_gen')}")
        self.prompter_gen = Prompter(model_type=config.get("llm_model_type_gen"), ollama_host=ollama_host)
        
        self.calculate_on = config.get("calculate_on")
        self.top_k = config.get("top_k")
        self.max_k = config.get("max_k")
        self.min_k = config.get("min_k")
        self.fusion_alpha = config.get("fusion_alpha", 0.5)
        
        with open(config.get("templates", {}).get("generative", "")) as f:
            self.generative_prompt = f.read()

        with open(config.get("templates", {}).get("extractive", "")) as f:
            self.extractive_prompt = f.read()
            
        self._logger.info("ObjectiveExtractor initialized with config: %s", config_path)
        
    def translate_ca_to_es(self, text: str) -> str:
        tokenized = self.ct2_tokenizer.tokenize(text)
        translated = self.ct2_translator.translate_batch([tokenized[0]])
        return self.ct2_tokenizer.detokenize(translated[0][0]['tokens'])
    
    def get_adaptive_top_k_from_combined(self, combined_nodes, max_k=10, min_k=3):

        if not combined_nodes:
            return []

        # Sort by score descending
        sorted_nodes = sorted(combined_nodes, key=lambda n: n.score or 0, reverse=True)
        
        scores = [n.score for n in sorted_nodes]
        self._logger.debug(f"Retrieved scores: {scores}")

        # Heuristic: stop adding if big drop in score (confidence decay)
        drop_threshold = 0.6  # relative drop
        top_nodes = [sorted_nodes[0]]

        for i in range(1, min(len(sorted_nodes), max_k)):
            prev_score = sorted_nodes[i - 1].score or 0
            curr_score = sorted_nodes[i].score or 0

            if curr_score / prev_score < drop_threshold:
                break
            top_nodes.append(sorted_nodes[i])

        # Enforce bounds
        if len(top_nodes) < min_k:
            top_nodes = sorted_nodes[:min(min_k, len(sorted_nodes))]

        return top_nodes
    
    def extract(self, text, option="generative"):
        try:
            doc = Document(text=text)
            nodes = self.node_parser.get_nodes_from_documents([doc])
            
            #top_k = min(self.top_k, len(nodes))

            # Setup retrievers
            vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model)
            vector_retriever = vector_index.as_retriever(similarity_top_k=self.top_k)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=self.top_k)

            # Combine results manually
            query = "objeto del contrato, objeto de la contratación, tiene por objeto, objetivos del contrato, objeto del pliego, objectivo"
            
            combined_nodes = self._combine_retrievers([bm25_retriever, vector_retriever], query, top_k=self.top_k)
            
            print(f"Combined nodes: {len(combined_nodes)} retrieved nodes")
            
            retrieved_nodes = self.get_adaptive_top_k_from_combined(
                combined_nodes, max_k=self.max_k, min_k=self.min_k, fusion_alpha=self.fusion_alpha
            )
            
            self._logger.info(f"Adaptive top-k nodes: {len(retrieved_nodes)}")
            
            scores = [n.score for n in retrieved_nodes if n.score is not None]
            print(scores)

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
                prompter = self.prompter_gen
            elif option == "extractive":
                prompt = self.extractive_prompt.format(context=context_joint)
                prompter = self.prompter_ex
            else:
                raise ValueError("Invalid option. Use 'generative' or 'extractive'.")
            result, _ = prompter.prompt(question=prompt, use_context=False)
            return result.strip()
        except Exception as e:
            return f"ERROR: {e}"

    # def _combine_retrievers(self, retrievers, query, top_k=4):
    #     all_nodes = []
    #     for retriever in retrievers:
    #         all_nodes.extend(retriever.retrieve(query))

    #     unique_nodes = {}
    #     for n in all_nodes:
    #         nid = n.node.node_id
    #         if nid not in unique_nodes or (n.score or 0) > (unique_nodes[nid].score or 0):
    #             unique_nodes[nid] = n

    #     return heapq.nlargest(top_k, unique_nodes.values(), key=lambda x: x.score or 0)
    
    def _combine_retrievers(self, retrievers, query, top_k=4, fusion_alpha=0.5):
        all_nodes = []
        source_map = {}  # track source by node_id

        for retriever in retrievers:
            name = type(retriever).__name__.lower()
            results = retriever.retrieve(query)
            all_nodes.extend(results)
            for n in results:
                source_map[n.node.node_id] = name

        # Separate scores by retriever
        bm25_nodes = [n for n in all_nodes if 'bm25' in source_map.get(n.node.node_id, '')]
        other_nodes = [n for n in all_nodes if 'bm25' not in source_map.get(n.node.node_id, '')]

        # Normalize BM25 scores (min-max)
        if bm25_nodes:
            bm25_scores = [n.score or 0 for n in bm25_nodes]
            min_s, max_s = min(bm25_scores), max(bm25_scores)
            for n in bm25_nodes:
                norm = (n.score - min_s) / (max_s - min_s + 1e-5) if max_s > min_s else 0
                n.score = norm

        # Combine by node ID
        combined = {}
        for n in bm25_nodes + other_nodes:
            nid = n.node.node_id
            if nid not in combined:
                combined[nid] = n
            else:
                existing = combined[nid]
                if fusion_alpha is not None:
                    # Blend scores (assume existing is from other retriever)
                    combined_score = fusion_alpha * (existing.score or 0) + (1 - fusion_alpha) * (n.score or 0)
                    existing.score = combined_score
                else:
                    # Pick whichever has higher score
                    if (n.score or 0) > (existing.score or 0):
                        combined[nid] = n

        final_nodes = list(combined.values())
        final_nodes = heapq.nlargest(top_k, final_nodes, key=lambda x: x.score or 0)

        self._logger.info(f"Combined scores: {[n.score for n in final_nodes]}")
        return final_nodes


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
    argparser.add_argument("--llm_model_type", type=str, default="llama3.1:8b", help="LLM model type to use for extraction if llm_model_type and llm_model_type_gen are not specified")
    argparser.add_argument("--llm_model_type_gen", type=str, default=None, help="LLM model type to use for generative extraction")
    argparser.add_argument("--llm_model_type_ex", type=str, default=None, help="LLM model type to use for extractive extraction")
    argparser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model to use for vectorization")
    argparser.add_argument("--top_k", type=int, default=20, help="Number of top results to retrieve from the vector store")
    argparser.add_argument("--fusion_alpha", type=float, default=0.5, help="Fusion alpha for combining scores from different retrievers")
    argparser.add_argument("--mode_extractive_generative", type=str, default="both", choices=["extractive", "generative", "both"], help="Mode of extraction: 'extractive', 'generative', or 'both'") 
    argparser.add_argument("--ollama_host", type=str, default="http://kumo01.tsc.uc3m.es:11434", help="Ollama host URL for LLM requests")
    
    args = argparser.parse_args()
    
    if args.llm_model_type_gen is None:
        llm_model_type_gen = args.llm_model_type
        print(f"Using default LLM model type for generative extraction: {llm_model_type_gen}")
    else:
        llm_model_type_gen = args.llm_model_type_gen
        print(f"Using LLM model type for generative extraction: {llm_model_type_gen}")
        
    if args.llm_model_type_ex is None:
        llm_model_type_ex = args.llm_model_type
        print(f"Using default LLM model type for extractive extraction: {llm_model_type_ex}")
    else:
        llm_model_type_ex = args.llm_model_type_ex
        print(f"Using LLM model type for extractive extraction: {llm_model_type_ex}")
    
    extractor = ObjectiveExtractor(
        config_path=pathlib.Path(args.config),
        ollama_host=args.ollama_host,
        calculate_on=args.calculate_on,
        llm_model_type_ex=llm_model_type_ex,
        llm_model_type_gen=llm_model_type_gen,
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
    df = df.sample(n=20, random_state=55)
    
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