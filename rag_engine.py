"""
rag_engine.py - Retrieval-Augmented Generation (RAG) Module for SugarcaneAI
Implements lightweight TF-IDF & Cosine Similarity vector retrieval over agronomic knowledge base,
coupled with grounded LLM answer generation using the Google Gemini SDK.
Gracefully operates in retrieval-only mode when no LLM API key is configured.
"""

import os
import glob
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()


class DocumentChunk:
    """Represents a single indexed knowledge chunk with traceable provenance."""

    def __init__(self, doc_name: str, section_title: str, text: str, chunk_id: int):
        self.doc_name = doc_name
        self.section_title = section_title
        self.text = text.strip()
        self.chunk_id = chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document": self.doc_name,
            "section": self.section_title,
            "text": self.text
        }


class SugarcaneRAGEngine:
    """
    Lightweight RAG Engine combining TF-IDF vector retrieval with Gemini LLM synthesis.
    Works 100% locally for vectorization and similarity matching.
    """

    def __init__(self, kb_dir: str = "knowledge_base"):
        self.kb_dir = os.path.abspath(kb_dir)
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.is_indexed: bool = False
        self.gemini_model_name: str = "gemini-3.8-flash"
        self._build_index()

    def reload_and_reindex(self, extra_texts: Optional[List[Tuple[str, str]]] = None) -> int:
        """
        Reloads knowledge base from disk, optionally ingesting uploaded user documents.
        extra_texts: List of (filename, content_str)
        """
        return self._build_index(extra_texts=extra_texts)

    def _build_index(self, extra_texts: Optional[List[Tuple[str, str]]] = None) -> int:
        """Parses documents, splits into overlapping chunks, and builds TF-IDF index."""
        self.chunks = []
        doc_files = []

        if os.path.exists(self.kb_dir):
            doc_files = glob.glob(os.path.join(self.kb_dir, "*.txt")) + glob.glob(os.path.join(self.kb_dir, "*.md"))

        chunk_counter = 0

        # Ingest files from knowledge_base directory
        for fpath in doc_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                file_chunks = self._chunk_text(content, fname, start_id=chunk_counter)
                self.chunks.extend(file_chunks)
                chunk_counter += len(file_chunks)
            except Exception as e:
                print(f"Warning: could not read {fpath}: {e}")

        # Ingest any dynamically uploaded documents
        if extra_texts:
            for fname, content in extra_texts:
                file_chunks = self._chunk_text(content, fname, start_id=chunk_counter)
                self.chunks.extend(file_chunks)
                chunk_counter += len(file_chunks)

        if not self.chunks:
            # Fallback starter chunk if directory is empty
            self.chunks.append(
                DocumentChunk(
                    doc_name="starter_notice.txt",
                    section_title="Initial Notice",
                    text="Sugarcane crop stand evaluation requires ground verification. Gaps over 50cm to 1m indicate potential crop loss.",
                    chunk_id=0
                )
            )

        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True
        return len(self.chunks)

    def _chunk_text(self, text: str, doc_name: str, start_id: int) -> List[DocumentChunk]:
        """Splits markdown/text into structured sections and sliding text chunks."""
        chunks: List[DocumentChunk] = []
        # Split on markdown headings or double line breaks
        sections = re.split(r'\n(?=#{1,3}\s+)', text)
        cid = start_id

        for sec in sections:
            lines = sec.strip().split('\n')
            if not lines:
                continue
            first_line = lines[0].strip()
            if first_line.startswith('#'):
                section_title = first_line.lstrip('#').strip()
                body = '\n'.join(lines[1:]).strip()
            else:
                section_title = "General Context"
                body = sec.strip()

            if not body:
                body = section_title

            # Word-based sliding window: ~250 words with 40-word overlap
            words = body.split()
            step = 200
            window_size = 260

            if len(words) <= window_size:
                chunks.append(DocumentChunk(doc_name, section_title, body, cid))
                cid += 1
            else:
                for i in range(0, len(words), step):
                    chunk_slice = " ".join(words[i : i + window_size])
                    if chunk_slice.strip():
                        chunks.append(DocumentChunk(doc_name, section_title, chunk_slice, cid))
                        cid += 1

        return chunks

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant knowledge chunks using cosine similarity on TF-IDF vectors.
        """
        if not self.is_indexed or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(sim_scores[idx])
            chunk = self.chunks[idx]
            results.append({
                "score": round(score, 4),
                "document": chunk.doc_name,
                "section": chunk.section_title,
                "text": chunk.text,
                "chunk_id": chunk.chunk_id
            })

        return results

    def query_assistant(
        self,
        query: str,
        api_key: Optional[str] = None,
        top_k: int = 3,
        min_relevance_threshold: float = 0.02
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Retrieve top-k passages via TF-IDF cosine similarity.
        2. Evaluate retrieval confidence score.
        3. If Gemini API key is available, generate grounded synthesis with citations.
        4. If no API key is available, return cleanly in retrieval-only mode.
        """
        retrieved_passages = self.retrieve(query, top_k=top_k)
        max_score = retrieved_passages[0]["score"] if retrieved_passages else 0.0

        # Assess retrieval confidence
        if max_score >= 0.12:
            confidence_level = "High"
        elif max_score >= 0.04:
            confidence_level = "Moderate"
        elif max_score >= min_relevance_threshold:
            confidence_level = "Low"
        else:
            confidence_level = "Insufficient Evidence"

        # Check if knowledge base genuinely contains relevant evidence
        if max_score < min_relevance_threshold:
            insufficient_msg = (
                "The agricultural knowledge base does not contain sufficient evidence to answer this question reliably. "
                f"(Highest retrieval similarity: {max_score:.4f}, below threshold {min_relevance_threshold}). "
                "To maintain strict agronomic safety and prevent hallucination, no speculative recommendation is made."
            )
            return {
                "answer": insufficient_msg,
                "retrieved_passages": retrieved_passages,
                "citations": [],
                "retrieval_confidence": confidence_level,
                "max_similarity_score": max_score,
                "mode": "insufficient_evidence",
                "error": None
            }

        # Format citations
        citations = []
        for p in retrieved_passages:
            if p["score"] >= min_relevance_threshold:
                c_str = f"[{p['document']} - Section: {p['section']}] (Similarity: {p['score']:.3f})"
                if c_str not in citations:
                    citations.append(c_str)

        # Check for Gemini API Key
        effective_key = api_key or os.getenv("GEMINI_API_KEY")
        if not effective_key:
            env_file = os.path.join(os.path.dirname(self.kb_dir), ".env")
            if os.path.exists(env_file):
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        env_raw = f.read().strip()
                        if env_raw.startswith("GEMINI_API_KEY="):
                            effective_key = env_raw.split("=", 1)[1].strip()
                        elif env_raw:
                            effective_key = env_raw
                except Exception:
                    pass

        if not effective_key:
            # Clean Retrieval-Only Mode
            retrieval_mode_text = (
                "**Retrieval-Only Mode Active** (No Gemini API Key provided).\n\n"
                "The system successfully identified relevant agronomic evidence from your local knowledge base. "
                "Review the retrieved passages and citations below. To enable synthesized LLM answers, "
                "enter a Google Gemini API key in the sidebar or `.env`."
            )
            return {
                "answer": retrieval_mode_text,
                "retrieved_passages": retrieved_passages,
                "citations": citations,
                "retrieval_confidence": confidence_level,
                "max_similarity_score": max_score,
                "mode": "retrieval_only",
                "error": None
            }

        # LLM Synthesis using official google-generativeai SDK
        try:
            import google.generativeai as genai
            genai.configure(api_key=effective_key)

            # Construct strict grounded prompt
            context_blocks = []
            for idx, p in enumerate(retrieved_passages):
                if p["score"] >= min_relevance_threshold:
                    context_blocks.append(
                        f"--- SOURCE [{idx+1}]: {p['document']} ({p['section']}) ---\n{p['text']}"
                    )
            context_text = "\n\n".join(context_blocks)

            system_instruction = (
                "You are the SugarcaneAI Agricultural Decision Support Assistant. "
                "Your role is to assist sugarcane growers, agronomists, and farm managers by providing "
                "strictly grounded, evidence-based recommendations on sugarcane gap management, stand establishment, "
                "and visual remote sensing limitations.\n\n"
                "CRITICAL RULES:\n"
                "1. Answer using ONLY the supplied agronomic evidence below. Do not fabricate facts, unverified treatments, or rates.\n"
                "2. Explicitly cite the document and section for every key recommendation (e.g., [sugarcane_guidelines.txt]).\n"
                "3. Never claim that an object-detected bounding box represents an exact plant count or subsurface disease cause.\n"
                "4. Always recommend ground scouting verification before major capital interventions.\n"
                "5. If the evidence is incomplete, clearly state the boundary of available knowledge."
            )

            prompt = (
                f"{system_instruction}\n\n"
                f"### RETRIEVED AGRONOMIC EVIDENCE:\n{context_text}\n\n"
                f"### USER INQUIRY:\n{query}\n\n"
                f"### GROUNDED ANSWER WITH CITATIONS:"
            )

            # Try modern Gemini models with fallback
            llm_response_text = ""
            candidate_errors = []
            for model_candidate in [
                "gemini-3.8-flash",
                "gemini-flash-latest",
                "gemini-2.5-flash-lite",
                "gemini-flash-lite-latest",
                "gemini-pro-latest"
            ]:
                try:
                    model = genai.GenerativeModel(model_candidate)
                    resp = model.generate_content(prompt)
                    if resp and resp.text:
                        llm_response_text = resp.text
                        break
                except Exception as model_err:
                    candidate_errors.append(f"{model_candidate}: {str(model_err)}")
                    continue

            if not llm_response_text:
                err_detail = " | ".join(candidate_errors[:2]) if candidate_errors else "empty response"
                raise RuntimeError(f"Gemini API models returned error: {err_detail}")

            return {
                "answer": llm_response_text,
                "retrieved_passages": retrieved_passages,
                "citations": citations,
                "retrieval_confidence": confidence_level,
                "max_similarity_score": max_score,
                "mode": "llm_grounded",
                "error": None
            }

        except Exception as e:
            fallback_answer = (
                f"**LLM Generation Notice:** Could not generate synthesis via Gemini API ({str(e)}). "
                "Falling back to verified retrieved evidence below."
            )
            return {
                "answer": fallback_answer,
                "retrieved_passages": retrieved_passages,
                "citations": citations,
                "retrieval_confidence": confidence_level,
                "max_similarity_score": max_score,
                "mode": "retrieval_fallback",
                "error": str(e)
            }
