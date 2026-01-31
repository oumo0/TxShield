import os
import pandas as pd
from typing import Dict, Any, List, Optional
from langchain_chroma import Chroma

from config.settings import CONFIG
from core.embeddings import JinaEmbedding
from utils.file_utils import load_system_input
from models.ernie_model import ErnieLLMWrapper, BaiduErnieAI
from global_instances import skip_rag_manager, exclusion_middleware, init_global_instances

init_global_instances()

class ContractAnalyzer:
    
    SIMILARITY_SEARCH_K = 5
    REFERENCE_TEXT_TRUNCATION_LENGTH = 500
    MIN_DOCUMENTS_FOR_RAG = 1
    
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self._initialize_components()
    
    def _initialize_components(self):

        skip_rag_manager.reset()
        embedding = JinaEmbedding()
        self.vector_store = self._load_vector_store(CONFIG.PERSIST_DIR, embedding)
        
        print(f"Vector store initialized: {self.vector_store._collection.count()} vectors")
        
        ernie_ai = BaiduErnieAI()
        self.llm = ErnieLLMWrapper(ernie_ai)
    
    def analyze(
        self,
        query: str,
        system_input_path: Optional[str] = None,
        system_input_text: Optional[str] = None
    ) -> Dict[str, Any]:

        try:
            system_input = self._prepare_system_input(system_input_path, system_input_text)
            
            input_data = {
                "input": query,
                "system_input": system_input,
                "system_input_path": system_input_path,
                "chat_history": [],
                "skip_rag": False
            }
            
            self._log_analysis_start(query, system_input_path, system_input)
            
            if system_input_path:
                self._analyze_exclusion_patterns(system_input_path)
            
            result = self._execute_core_analysis(input_data)
            
            return result
            
        except Exception as e:
            print(f"Analysis pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Analysis pipeline error: {str(e)}",
                "status": "failed",
                "similarity_info": None,
                "used_rag": False
            }
    
    def _prepare_system_input(
        self,
        system_input_path: Optional[str],
        system_input_text: Optional[str]
    ) -> str:

        if system_input_path:
            return load_system_input(system_input_path)
        elif system_input_text:
            return system_input_text
        else:
            return ""
    
    def _log_analysis_start(
        self,
        query: str,
        system_input_path: Optional[str],
        system_input: str
    ):
        
        print("\n" + "=" * 60)
        print("Contract Analysis Pipeline Initiated")
        print(f"Query: {query}")
        print(f"System Input Source: {system_input_path or 'Text Input'}")
        print(f"System Input Length: {len(system_input)}")
        print("=" * 60 + "\n")
    
    def _analyze_exclusion_patterns(self, system_input_path: str):
        print("\nExclusion Middleware Analysis:")
        exclusion_info = exclusion_middleware.get_contract_info(system_input_path)
        
        analysis_summary = [
            f"Contract Name: {exclusion_info['contract_name']}",
            f"Transaction Hash: {exclusion_info['transaction_hash'][:20]}...",
            f"Knowledge Base: {exclusion_info['kb_file']}",
            f"Exclusion Rules Present: {exclusion_info['has_exclusion_rules']}"
        ]
        
        for line in analysis_summary:
            print(f"  {line}")
    
    def _execute_core_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:

        print("Retrieving relevant documents...")
        
        query_text = input_data.get("system_input", "") or input_data.get("input", "")
        retrieved_docs = self._semantic_retrieval(query_text)
        filtered_docs = self._apply_document_filters(retrieved_docs, input_data)
        use_rag = self._determine_rag_strategy(filtered_docs)
        skip_rag_manager.set_skip_rag(not use_rag)
        
        llm_input = self._prepare_llm_input(input_data, filtered_docs, use_rag)
        
        print("Executing LLM analysis...")
        llm_response = self.llm.invoke(llm_input)
        
        return self._construct_analysis_results(
            llm_response, 
            filtered_docs, 
            use_rag,
            retrieved_docs.get("similarity_scores", [])
        )
    
    def _semantic_retrieval(self, query_text: str) -> Dict[str, Any]:

        docs_with_scores = self.vector_store.similarity_search_with_score(
            query_text,
            k=self.SIMILARITY_SEARCH_K
        )
        
        docs = [doc for doc, score in docs_with_scores]
        similarity_scores = [score for doc, score in docs_with_scores]
        
        print(f"Initial retrieval results: {len(docs)} documents")
        
        return {
            "documents": docs,
            "similarity_scores": similarity_scores,
            "documents_with_scores": docs_with_scores
        }
    
    def _apply_document_filters(
        self,
        retrieval_results: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> List:

        docs = retrieval_results["documents"]
        docs_with_scores = retrieval_results["documents_with_scores"]
        similarity_scores = retrieval_results["similarity_scores"]
        
        test_file_path = input_data.get("system_input_path", "")
        if test_file_path and docs:
            docs = exclusion_middleware.filter_documents(docs, test_file_path)
            print(f"After exclusion filtering: {len(docs)} documents")
        
        threshold = CONFIG.SIMILARITY_THRESHOLD
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in docs_with_scores:
            if doc in docs and score <= threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)
                self._enhance_document_metadata(doc, score)
        
        print(f"After similarity filtering: {len(filtered_docs)} documents")
        
        return filtered_docs
    
    def _enhance_document_metadata(self, document, similarity_score: float):
        """Enhance document with additional metadata"""
        if not hasattr(document, 'metadata') or document.metadata is None:
            document.metadata = {}
        elif not isinstance(document.metadata, dict):
            document.metadata = dict(document.metadata)
        
        document.metadata['similarity_score'] = float(similarity_score)
        document.metadata['retrieval_confidence'] = 1.0 - similarity_score
    
    def _determine_rag_strategy(self, filtered_docs: List) -> bool:

        return len(filtered_docs) >= self.MIN_DOCUMENTS_FOR_RAG
    
    def _prepare_llm_input(
        self,
        input_data: Dict[str, Any],
        filtered_docs: List,
        use_rag: bool
    ) -> Dict[str, Any]:

        reference_text = ""
        if use_rag and filtered_docs:
            reference_text = self._construct_contextual_reference(filtered_docs)
        
        return {
            "query": input_data.get("input", ""),
            "system_input": input_data.get("system_input", ""),
            "reference_text": reference_text,
            "skip_rag": not use_rag,
            "prompt_type": "rag" if use_rag else "direct",
            "contextual_documents": len(filtered_docs)
        }
    
    def _construct_contextual_reference(self, filtered_docs: List) -> str:

        reference_segments = []
        
        for i, doc in enumerate(filtered_docs):
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                source = doc.metadata.get('source_file', 'Unknown Source') if hasattr(doc, 'metadata') else 'Unknown Source'
                
                segment = [
                    f"[Document {i+1}]",
                    f"Source: {source}",
                    f"Content: {content[:self.REFERENCE_TEXT_TRUNCATION_LENGTH]}...",
                    "------"
                ]
                
                reference_segments.append("\n".join(segment))
        
        return "\n\n".join(reference_segments) if reference_segments else "No relevant reference documents found"
    
    def _construct_analysis_results(
        self,
        llm_response: Any,
        filtered_docs: List,
        use_rag: bool,
        similarity_scores: List[float]
    ) -> Dict[str, Any]:

        result = {
            "answer": llm_response.content if hasattr(llm_response, 'content') else str(llm_response),
            "used_rag": use_rag,
            "retrieved_docs_count": len(filtered_docs),
            "analysis_strategy": "rag" if use_rag else "direct"
        }
        
        if similarity_scores:
            result["similarity_analysis"] = self._analyze_similarity_distribution(similarity_scores)
        
        if hasattr(llm_response, 'additional_kwargs'):
            metadata_keys = ["_full_prompt_content", "_api_messages", "reasoning_trace"]
            for key in metadata_keys:
                if key in llm_response.additional_kwargs:
                    result[key] = llm_response.additional_kwargs[key]
        
        return result
    
    def _analyze_similarity_distribution(self, similarity_scores: List[float]) -> Dict[str, Any]:

        if not similarity_scores:
            return {}
        
        import numpy as np
        
        scores_array = np.array(similarity_scores)
        
        return {
            "statistical_summary": {
                "minimum": float(np.min(scores_array)),
                "maximum": float(np.max(scores_array)),
                "mean": float(np.mean(scores_array)),
                "median": float(np.median(scores_array)),
                "standard_deviation": float(np.std(scores_array))
            },
            "distribution_metrics": {
                "score_range": float(np.max(scores_array) - np.min(scores_array)),
                "variance": float(np.var(scores_array)),
                "confidence_interval": self._calculate_confidence_interval(scores_array)
            },
            "sample_count": len(similarity_scores)
        }
    
    def _calculate_confidence_interval(self, scores_array) -> Dict[str, float]:

        import numpy as np
        
        confidence_level = 0.95
        n = len(scores_array)
        
        if n < 2:
            return {"lower": 0.0, "upper": 0.0}
        
        mean = np.mean(scores_array)
        std_err = np.std(scores_array, ddof=1) / np.sqrt(n)
        
        if n < 30:
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin = t_value * std_err
        else:
            from scipy import stats
            z_value = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_value * std_err
        
        return {
            "lower": float(max(0.0, mean - margin)),
            "upper": float(min(1.0, mean + margin)),
            "confidence_level": confidence_level
        }
    
    def _load_vector_store(self, persist_dir: str, embedding: JinaEmbedding) -> Chroma:
        from core.vector_store import load_or_create_vector_store as _load_or_create_vector_store
        return _load_or_create_vector_store(persist_dir, embedding)


def analyze_contract(
    query: str,
    system_input_path: Optional[str] = None,
    system_input_text: Optional[str] = None
) -> Dict[str, Any]:

    analyzer = ContractAnalyzer()
    return analyzer.analyze(query, system_input_path, system_input_text)