import os
import traceback
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .settings import CONFIG
from .file_utils import load_system_input, load_or_create_vector_store
from .embedding_model import JinaEmbedding
from .ernie_model import BaiduErnieAI, ErnieLLMWrapper
from .base_chain import BaseChain
from .rag_chain import RAGEnhancementChain as RAGChain
from .router import PromptRouter as Router
from .debug_utils import get_skip_rag_manager

@dataclass
class AnalysisConfig:
    """Configuration for contract analysis pipeline"""
    persist_dir: str = CONFIG["PERSIST_DIR"]
    csv_path: str = CONFIG["CSV_PATH"]
    enable_monitoring: bool = True

@dataclass
class AnalysisResult:
    """Structured result from contract analysis"""
    answer: str
    context: Any
    used_rag: bool
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        result = {
            "answer": self.answer,
            "context": self.context,
            "used_rag": self.used_rag
        }
        if self.metadata:
            result.update(self.metadata)
        return result

class ContractAnalyzer:
    """Main orchestrator for smart contract security analysis"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        self.skip_rag_manager = get_skip_rag_manager()
        self.embedding = JinaEmbedding()
        self.vector_store = load_or_create_vector_store(
            self.config.persist_dir, 
            self.embedding
        )
        self.llm = ErnieLLMWrapper(BaiduErnieAI())
        
        print(f"✅ Vector store initialized: {self.vector_store._collection.count()} vectors")
        
    def _prepare_input_data(
        self, 
        query: str, 
        system_input_path: Optional[str] = None,
        system_input_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare structured input for analysis pipeline"""
        system_input = ""
        if system_input_path:
            system_input = load_system_input(system_input_path)
        elif system_input_text:
            system_input = system_input_text
            
        return {
            "input": query,
            "system_input": system_input,
            "system_input_path": system_input_path,
            "chat_history": [],
            "skip_rag": False
        }
    
    def _execute_pipeline(self, input_data: Dict[str, Any]) -> AnalysisResult:
        """Execute the complete analysis pipeline"""
        # Reset state
        self.skip_rag_manager.reset()
        
        # Initialize chains
        base_chain = BaseChain(
            self.vector_store, 
            self.llm, 
            self.config.csv_path, 
            input_data
        )
        rag_chain = RAGChain(
            self.vector_store, 
            self.llm, 
            self.config.csv_path, 
            input_data
        )
        router = Router()
        
        # Step 1: Context retrieval
        context = base_chain.enhanced_retriever(input_data)
        
        # Step 2: Intelligent routing
        routed_data = rag_chain.route_based_on_rag({
            **input_data,
            "context": context
        })
        
        # Step 3: Prompt selection
        llm_input = router.route_to_appropriate_prompt(routed_data)
        
        if "error" in llm_input:
            raise ValueError(f"Prompt routing failed: {llm_input['error']}")
        
        # Step 4: LLM inference
        llm_response = self.llm.invoke(llm_input)
        
        # Prepare metadata
        metadata = {}
        if hasattr(llm_response, 'additional_kwargs'):
            for key in ["_full_prompt_content", "_api_messages"]:
                if key in llm_response.additional_kwargs:
                    metadata[key] = llm_response.additional_kwargs[key]
        
        return AnalysisResult(
            answer=llm_response.content if hasattr(llm_response, 'content') else str(llm_response),
            context=context,
            used_rag=not self.skip_rag_manager.get_skip_rag(),
            metadata=metadata
        )
    
    def analyze(
        self, 
        query: str, 
        system_input_path: Optional[str] = None,
        system_input_text: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Main analysis entry point
        
        Args:
            query: Analysis query
            system_input_path: Path to system input file
            system_input_text: Raw system input text
            verbose: Enable detailed logging
            
        Returns:
            Analysis result dictionary
        """
        try:
            if verbose:
                print("=" * 50)
                print("Smart Contract Analysis Pipeline")
                print(f"Query: {query[:100]}...")
                print(f"System input: {system_input_path or 'Text input'}")
                print("=" * 50)
            
            # Prepare input
            input_data = self._prepare_input_data(
                query, 
                system_input_path, 
                system_input_text
            )
            
            # Execute pipeline
            result = self._execute_pipeline(input_data)
            
            if verbose:
                print(f"Analysis complete")
                print(f"RAG utilized: {result.used_rag}")
                print(f"Context documents: {len(result.context)}")
                print("=" * 50)
            
            return result.to_dict()
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            if verbose:
                traceback.print_exc()
            
            return {
                "error": f"Analysis failed: {str(e)}",
                "status": "failed",
                "used_rag": False
            }
        
def create_analyzer(config: AnalysisConfig = None) -> ContractAnalyzer:
    return ContractAnalyzer(config)

def demonstrate_analysis():
    config = AnalysisConfig()
    analyzer = create_analyzer(config)
    
    # Example query and input
    test_query = "Please analyze this transaction to determine if it's an attack."
    test_path = "path/to/contract_analysis_input.json"
    
    print("Starting demonstration...")
    result = analyzer.analyze(
        query=test_query,
        system_input_path=test_path,
        verbose=True
    )
    
    # Display results
    if "error" not in result:
        print("\nAnalysis Results:")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"RAG Used: {result['used_rag']}")
        print(f"Context Documents: {len(result.get('context', []))}")
    else:
        print(f"Analysis failed: {result['error']}")
    
    return result

if __name__ == "__main__":
    demonstrate_analysis()