from typing import List, Tuple, Dict
from langchain.schema import Document
from tqdm import tqdm
import numpy as np

class RetrievalEvaluator:

    @staticmethod
    def compute_exact_match(results: List[Document], target: Document) -> int:
        return int(any(doc.page_content.strip() == target.page_content.strip() for doc in results))

    @staticmethod
    def compute_precision_at_k(results: List[Document], target: Document, k: int) -> float:
        hits = sum(1 for doc in results[:k] if doc.page_content.strip() == target.page_content.strip())
        return hits / k

    @staticmethod
    def compute_recall_at_k(results: List[Document], target: Document, total_relevant: int = 1) -> float:
        hits = sum(1 for doc in results if doc.page_content.strip() == target.page_content.strip())
        return hits / total_relevant

    @staticmethod
    def compute_mrr(results: List[Document], target: Document) -> float:
        for idx, doc in enumerate(results, 1):
            if doc.page_content.strip() == target.page_content.strip():
                return 1.0 / idx
        return 0.0

    @staticmethod
    def evaluate_knexus_query(
        knexus_mngr,
        target_doc: Document,
        query_text: str,
        k: int = 5,
        min_score: float = 0.7,
        verbose: bool = False
    ) -> Dict[str, float]:
        docs, scores = knexus_mngr.query_similarity_search(
            query_text,
            verbose=verbose,
            min_score=min_score,
            mode="min",
            reverse=False
        )
    
        top_k_docs = docs[:k]
    
        return {
            "exact_match": compute_exact_match(top_k_docs, target_doc),
            "precision@k": compute_precision_at_k(top_k_docs, target_doc, k),
            "recall@k": compute_recall_at_k(top_k_docs, target_doc),
            "mrr": compute_mrr(top_k_docs, target_doc)
        }

        # Reuse metric methods (assumed already defined)
    @staticmethod
    def evaluate_knexus_batch(
        knexus_mngr,
        queries: list,
        docs: List[Document],
        k: int = 5,
        min_score: float = 0.7,
        verbose: bool = False,
        precision: int = 4,
    ) -> Dict[str, float]:
        exact_matches = []
        precisions = []
        recalls = []
        mrrs = []
    
        for idx in tqdm(range(len(docs)), desc="Evaluating retrievals"):
            target_doc = docs[idx]
            query_text = queries[idx]
    
            try:
                retrieved_docs, _ = knexus_mngr.query_similarity_search(
                    query_text,
                    verbose=verbose,
                    min_score=min_score,
                    reverse=False
                )
            except Exception as e:
                print(f"[Query Error] Index {idx}: {e}")
                continue
    
            # Use previously defined evaluation methods
            exact = compute_exact_match(retrieved_docs[:k], target_doc)
            precision = compute_precision_at_k(retrieved_docs, target_doc, k)
            recall = compute_recall_at_k(retrieved_docs, target_doc)
            mrr = compute_mrr(retrieved_docs[:k], target_doc)
    
            exact_matches.append(exact)
            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(mrr)
    
        # Aggregate the metrics
        return {
            "exact_match": round(np.mean(exact_matches), precision) if exact_matches else 0.0,
            "precision@k": round(np.mean(precisions), precision) if precisions else 0.0,
            "recall@k": round(np.mean(recalls), precision) if recalls else 0.0,
            "mrr": round(np.mean(mrrs), precision) if mrrs else 0.0
        }
    
    # Reuse metric methods (assumed already defined)
    @staticmethod
    def evaluate_knexus_batch_scope(
        docs: List[Document],
        scope_helper,
        knexus_mngr,
        k: int = 5,
        min_score: float = 0.7,
        verbose: bool = False
    ) -> Dict[str, float]:
        exact_matches = []
        precisions = []
        recalls = []
        mrrs = []
    
        for idx in tqdm(range(len(docs)), desc="Evaluating retrievals"):
            target_doc = docs[idx]
            try:
                query_text = scope_helper.pull_sow_prompt(target_doc.page_content)
            except Exception as e:
                print(f"[Prompt Error] Index {idx}: {e}")
                continue
    
            try:
                retrieved_docs, _ = knexus_mngr.query_similarity_search(
                    query_text,
                    verbose=verbose,
                    min_score=min_score,
                    mode="min",
                    reverse=False
                )
            except Exception as e:
                print(f"[Query Error] Index {idx}: {e}")
                continue
    
            # Use previously defined evaluation methods
            exact = compute_exact_match(retrieved_docs[:k], target_doc)
            precision = compute_precision_at_k(retrieved_docs, target_doc, k)
            recall = compute_recall_at_k(retrieved_docs, target_doc)
            mrr = compute_mrr(retrieved_docs[:k], target_doc)
    
            exact_matches.append(exact)
            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(mrr)
    
        # Aggregate the metrics
        return {
            "exact_match": round(np.mean(exact_matches), 4) if exact_matches else 0.0,
            "precision@k": round(np.mean(precisions), 4) if precisions else 0.0,
            "recall@k": round(np.mean(recalls), 4) if recalls else 0.0,
            "mrr": round(np.mean(mrrs), 4) if mrrs else 0.0
        }