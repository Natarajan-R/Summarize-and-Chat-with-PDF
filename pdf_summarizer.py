import os
from typing import Union, Dict, Optional, List, Tuple  # Add this at the top with other imports

# Force offline mode to use local models only
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import requests
import json
from pathlib import Path
import PyPDF2
import pdfplumber
from datetime import datetime
import argparse
import sys
import re
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
import tiktoken  # For accurate token counting
import sqlite3
import pickle
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # For vector similarity search
from json import JSONEncoder


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def custom_json_dumps(obj):
    return json.dumps(obj, cls=CustomJSONEncoder)

@dataclass
class DocumentChunk:
    """Class to represent a document chunk with metadata"""
    chunk_id: str
    content: str
    chunk_index: int
    tokens: int
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

@dataclass
class DocumentSession:
    """Class to represent a persistent document session"""
    session_id: str
    pdf_path: str
    pdf_hash: str
    created_at: str
    updated_at: str
    total_chunks: int
    model_name: str
    chunk_tokens: int
    overlap_tokens: int
    summary: Optional[str] = None
    stats: Optional[Dict] = None

@dataclass
class QAResult:
    """Class to hold Q&A results"""
    question: str
    answer: str
    confidence: str  # "high", "medium", "low"
    source_chunks: List[int]  # indices of chunks that contributed to answer
    processing_time: float
   
    # 3. Fix the QAResult __str__ method
    def __str__(self):
        return f"Q: {self.question}\nA: {self.answer}\nConfidence: {self.confidence}\nProcessing time: {self.processing_time:.2f}s"

@dataclass
class DocumentStats:
    """Class to hold document statistics"""
    pages: int = 0
    characters: int = 0
    words: int = 0
    sentences: int = 0
    paragraphs: int = 0
    
    def __str__(self):
        return f"Pages: {self.pages}, Words: {self.words:,}, Characters: {self.characters:,}, Sentences: {self.sentences}, Paragraphs: {self.paragraphs}"


@dataclass
class SummaryStats:
    """Class to hold summary statistics and compression ratios"""
    original: Union[DocumentStats, dict]
    summary: Union[DocumentStats, dict]
    compression_ratio_words: float = 0.0
    compression_ratio_chars: float = 0.0
    processing_time: float = 0.0
    
    def calculate_ratios(self):
        """Calculate compression ratios with support for both objects and dicts"""
        original_words = self.original.words if hasattr(self.original, 'words') else self.original.get('words', 0)
        summary_words = self.summary.words if hasattr(self.summary, 'words') else self.summary.get('words', 0)
        original_chars = self.original.characters if hasattr(self.original, 'characters') else self.original.get('characters', 0)
        summary_chars = self.summary.characters if hasattr(self.summary, 'characters') else self.summary.get('characters', 0)
        
        if original_words > 0:
            self.compression_ratio_words = (summary_words / original_words) * 100
        if original_chars > 0:
            self.compression_ratio_chars = (summary_chars / original_chars) * 100
    
    def get_summary_report(self) -> str:
        """Generate a formatted statistics report with dict/object support"""
        self.calculate_ratios()
        
        # Helper function to get stats safely
        def get_stat(stat, field, default='N/A'):
            if hasattr(stat, field):
                return getattr(stat, field)
            return stat.get(field, default)
        
        original = self.original
        summary = self.summary
        
        report = f"""
## Document Statistics

### Original Document
- **Pages:** {get_stat(original, 'pages')}
- **Words:** {get_stat(original, 'words', 0):,}
- **Characters:** {get_stat(original, 'characters', 0):,}
- **Sentences:** {get_stat(original, 'sentences')}
- **Paragraphs:** {get_stat(original, 'paragraphs')}

### Generated Summary
- **Words:** {get_stat(summary, 'words', 0):,}
- **Characters:** {get_stat(summary, 'characters', 0):,}
- **Sentences:** {get_stat(summary, 'sentences')}
- **Paragraphs:** {get_stat(summary, 'paragraphs')}

### Compression Analysis
- **Word Compression:** {self.compression_ratio_words:.1f}% of original
- **Character Compression:** {self.compression_ratio_chars:.1f}% of original
- **Word Reduction:** {100 - self.compression_ratio_words:.1f}%
- **Processing Time:** {self.processing_time:.2f} seconds
"""
        return report


class VectorDatabase:
    """Vector database for storing and searching document chunks"""
    
    def __init__(self, db_path: str = "pdf_analyzer.db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector database
        
        Args:
            db_path: Path to SQLite database file
            embedding_model: Sentence transformer model for embeddings
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        
                # Initialize embedding model from local setup
        print(f"Loading local embedding model: {embedding_model}")
        try:
            # Set offline mode
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            
            self.embedding_model = SentenceTransformer(
                embedding_model
            )
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Successfully loaded local model ({self.embedding_dim}D)")
            
        except Exception as e:
            print(f"Could not load local embedding model: {e}")
            print("Falling back to keyword-based search...")
            self.embedding_model = None
            self.embedding_dim = 384

        
        # Initialize database
        self.init_database()
        
        # FAISS index for similarity search
        self.faiss_index = None
        self.chunk_ids = []  # To map FAISS indices to chunk IDs
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    pdf_path TEXT NOT NULL,
                    pdf_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    chunk_tokens INTEGER NOT NULL,
                    overlap_tokens INTEGER NOT NULL,
                    summary TEXT,
                    stats TEXT
                )
            ''')
            
            # Chunks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    tokens INTEGER NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Q&A history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_history (
                    qa_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    source_chunks TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.commit()
    
    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file for change detection"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def generate_session_id(self, pdf_path: str, pdf_hash: str) -> str:
        """Generate unique session ID based on file path and hash"""
        combined = f"{pdf_path}_{pdf_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def session_exists(self, pdf_path: str) -> Optional[str]:
        """Check if session exists for this PDF file"""
        if not os.path.exists(pdf_path):
            return None
        
        pdf_hash = self.compute_file_hash(pdf_path)
        session_id = self.generate_session_id(pdf_path, pdf_hash)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id FROM sessions WHERE session_id = ? AND pdf_hash = ?",
                (session_id, pdf_hash)
            )
            result = cursor.fetchone()
            return session_id if result else None

    def save_session(self, session: DocumentSession, chunks: List[DocumentChunk]):
        """Save document session and chunks to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert all objects to serializable format
            stats_data = None
            if session.stats:
                stats_data = custom_json_dumps(session.stats)
            
            # Save session
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, pdf_path, pdf_hash, created_at, updated_at, total_chunks, 
                model_name, chunk_tokens, overlap_tokens, summary, stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id, session.pdf_path, session.pdf_hash,
                session.created_at, session.updated_at, session.total_chunks,
                session.model_name, session.chunk_tokens, session.overlap_tokens,
                session.summary, stats_data
            ))
            
            # Save chunks
            for chunk in chunks:
                embedding_blob = pickle.dumps(chunk.embedding) if chunk.embedding is not None else None
                metadata = custom_json_dumps(chunk.metadata) if chunk.metadata else None
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, session_id, content, chunk_index, tokens, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk.chunk_id, session.session_id, chunk.content, chunk.chunk_index,
                    chunk.tokens, embedding_blob, metadata
                ))
            
            conn.commit()

    
    def load_session(self, session_id: str) -> Tuple[Optional[DocumentSession], List[DocumentChunk]]:
        """Load document session and chunks from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load session
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()
            
            if not session_row:
                return None, []
            
            session = DocumentSession(
                session_id=session_row[0],
                pdf_path=session_row[1],
                pdf_hash=session_row[2],
                created_at=session_row[3],
                updated_at=session_row[4],
                total_chunks=session_row[5],
                model_name=session_row[6],
                chunk_tokens=session_row[7],
                overlap_tokens=session_row[8],
                summary=session_row[9],
                stats=json.loads(session_row[10]) if session_row[10] else None
            )
            
            # Load chunks
            cursor.execute(
                "SELECT * FROM chunks WHERE session_id = ? ORDER BY chunk_index",
                (session_id,)
            )
            chunk_rows = cursor.fetchall()
            
            chunks = []
            for row in chunk_rows:
                embedding = pickle.loads(row[5]) if row[5] else None
                chunk = DocumentChunk(
                    chunk_id=row[0],
                    content=row[2],
                    chunk_index=row[3],
                    tokens=row[4],
                    embedding=embedding,
                    metadata=json.loads(row[6]) if row[6] else None
                )
                chunks.append(chunk)
            
            return session, chunks
    
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for list of texts"""
        if not self.embedding_model:
            # Return dummy embeddings if model not available
            return [np.random.rand(self.embedding_dim) for _ in texts]
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return [np.random.rand(self.embedding_dim) for _ in texts]
    
    def build_faiss_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index for fast similarity search"""
        if not chunks:
            return
        
        embeddings = []
        self.chunk_ids = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                self.chunk_ids.append(chunk.chunk_id)
        
        if embeddings:
            embeddings_array = np.vstack(embeddings).astype('float32')
            
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.faiss_index.add(embeddings_array)
            
            print(f"Built FAISS index with {len(embeddings)} chunks")
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar chunks using vector similarity"""
        if not self.faiss_index or not self.embedding_model:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            similarities, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.chunk_ids)))
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[idx]
                    results.append((chunk_id, float(similarity)))
            
            return results
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def save_qa_result(self, session_id: str, qa_result: QAResult):
        """Save Q&A result to history"""
        qa_id = hashlib.md5(f"{session_id}_{qa_result.question}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO qa_history 
                (qa_id, session_id, question, answer, confidence, source_chunks, processing_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                qa_id, session_id, qa_result.question, qa_result.answer,
                qa_result.confidence, json.dumps(qa_result.source_chunks),
                qa_result.processing_time, datetime.now().isoformat()
            ))
            conn.commit()
    
    def get_qa_history(self, session_id: str, limit: int = 20) -> List[QAResult]:
        """Get Q&A history for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT question, answer, confidence, source_chunks, processing_time, created_at
                FROM qa_history 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (session_id, limit))
            
            results = []
            for row in cursor.fetchall():
                qa_result = QAResult(
                    question=row[0],
                    answer=row[1],
                    confidence=row[2],
                    source_chunks=json.loads(row[3]),
                    processing_time=row[4]
                )
                results.append(qa_result)
            
            return results
    
    def list_sessions(self) -> List[Dict]:
        """List all available sessions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, pdf_path, created_at, updated_at, total_chunks, model_name
                FROM sessions 
                ORDER BY updated_at DESC
            ''')
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'session_id': row[0],
                    'pdf_path': row[1],
                    'created_at': row[2],
                    'updated_at': row[3],
                    'total_chunks': row[4],
                    'model_name': row[5]
                })
            
            return sessions
    
    def delete_session(self, session_id: str):
        """Delete a session and all associated data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM qa_history WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def get_session_stats(self, session_id):
        """Get statistics for a specific session"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT statistics FROM sessions 
                WHERE session_id = ?
            """, (session_id,))
            
            result = cursor.fetchone()
            if result and result[0]:
                # Parse the JSON statistics
                stats_json = result[0]
                if isinstance(stats_json, str):
                    import json
                    stats = json.loads(stats_json)
                else:
                    stats = stats_json
                return stats
            return None
        except Exception as e:
            print(f"Error getting session stats: {e}")
            return None


class PDFSummarizer:
    def __init__(self, ollama_url="http://localhost:11434", model_name="mistral", 
                 max_chunk_tokens=3000, overlap_tokens=200, db_path="pdf_analyzer.db"):
        """
        Initialize the PDF Summarizer with vector database persistence
        
        Args:
            ollama_url (str): URL of the Ollama API endpoint
            model_name (str): Name of the model to use (e.g., 'mistral')
            max_chunk_tokens (int): Maximum tokens per chunk (adjust based on your model's context)
            overlap_tokens (int): Number of tokens to overlap between chunks for context preservation
            db_path (str): Path to vector database file
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        # Chunking configuration
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize vector database
        self.vector_db = VectorDatabase(db_path)
        
        # Current session data
        self.current_session = None
        self.document_chunks = []
        self.document_chunk_objects = []  # List of DocumentChunk objects
        self.document_path = None
        self.document_stats = None
        
        # Initialize tokenizer (using GPT-4 tokenizer as approximation)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            print("Warning: tiktoken not available, falling back to character-based estimation")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text accurately"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters for English
            return len(text) // 4


    def intelligent_text_splitter(self, text: str) -> List[str]:
        """
        Split text intelligently based on content structure and token limits
        
        Args:
            text (str): Input text to split
            
        Returns:
            List[str]: List of intelligently chunked text pieces
        """
        # If text is short enough, return as single chunk
        if self.count_tokens(text) <= self.max_chunk_tokens:
            return [text]
        
        chunks = []
        
        # Step 1: Try to split by major sections/chapters
        major_sections = self.split_by_sections(text)
        
        for section in major_sections:
            section_tokens = self.count_tokens(section)
            
            if section_tokens <= self.max_chunk_tokens:
                # Section fits in one chunk
                chunks.append(section)
            else:
                # Section too large, split by paragraphs
                paragraph_chunks = self.split_by_paragraphs(section)
                chunks.extend(paragraph_chunks)
        
        # Step 2: Add overlap between chunks for context preservation
        overlapped_chunks = self.add_overlap(chunks)
        
        return overlapped_chunks



        
    def load_or_create_session(self, pdf_path: str) -> str:
        """
        Load existing session or create new one for PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            session_id: ID of loaded or created session
        """
        # Check if session exists
        session_id = self.vector_db.session_exists(pdf_path)
        
        if session_id:
            print(f"Loading existing session: {session_id}")
            session, chunks = self.vector_db.load_session(session_id)
            
            if session and chunks:
                self.current_session = session
                self.document_path = session.pdf_path
                self.document_chunks = [chunk.content for chunk in chunks]
                self.document_chunk_objects = chunks
                
                # Rebuild FAISS index
                self.vector_db.build_faiss_index(chunks)
                
                print(f"Loaded session with {len(chunks)} chunks")
                return session_id
            else:
                print("Session data corrupted, creating new session")
        
        # Create new session
        pdf_hash = self.vector_db.compute_file_hash(pdf_path)
        session_id = self.vector_db.generate_session_id(pdf_path, pdf_hash)
        
        self.current_session = DocumentSession(
            session_id=session_id,
            pdf_path=pdf_path,
            pdf_hash=pdf_hash,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            total_chunks=0,
            model_name=self.model_name,
            chunk_tokens=self.max_chunk_tokens,
            overlap_tokens=self.overlap_tokens
        )
        
        print(f"Created new session: {session_id}")
        return session_id

    def save_current_session(self, final_summary: str = None, stats: Union[dict, SummaryStats] = None):
        """Save current session to database"""
        if not self.current_session:
            return
        
        self.current_session.updated_at = datetime.now().isoformat()
        self.current_session.total_chunks = len(self.document_chunk_objects)
        if final_summary:
            self.current_session.summary = final_summary
        if stats:
            # Handle both dict and SummaryStats
            if isinstance(stats, SummaryStats):
                self.current_session.stats = {
                    'original': stats.original.__dict__ if hasattr(stats.original, '__dict__') else stats.original,
                    'summary': stats.summary.__dict__ if hasattr(stats.summary, '__dict__') else stats.summary,
                    'processing_time': stats.processing_time
                }
            else:
                self.current_session.stats = stats
        
        self.vector_db.save_session(self.current_session, self.document_chunk_objects)
        print(f"Session saved: {len(self.document_chunk_objects)} chunks persisted")


    
    def create_chunk_objects(self, text_chunks: List[str]) -> List[DocumentChunk]:
        """Convert text chunks to DocumentChunk objects with embeddings"""
        chunk_objects = []
        
        print("Creating embeddings for document chunks...")
        
        # Create embeddings for all chunks
        if self.vector_db.embedding_model:
            embeddings = self.vector_db.create_embeddings(text_chunks)
        else:
            embeddings = [None] * len(text_chunks)
        
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_id = f"{self.current_session.session_id}_chunk_{i}"
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                chunk_index=i,
                tokens=self.count_tokens(chunk_text),
                embedding=embedding,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'model': self.model_name
                }
            )
            chunk_objects.append(chunk)
        
        return chunk_objects
    
    def find_relevant_chunks(self, question: str, top_k: int = 3) -> List[Tuple[int, str, float]]:
        """
        Find the most relevant chunks using vector similarity and keyword matching
        
        Args:
            question (str): The question to answer
            top_k (int): Number of top chunks to return
            
        Returns:
            List of tuples: (chunk_index, chunk_text, relevance_score)
        """
        if not self.document_chunk_objects:
            return []
        
        # Try vector search first
        vector_results = self.vector_db.vector_search(question, top_k * 2)  # Get more for reranking
        
        if vector_results:
            print(f"Using vector similarity search")
            results = []
            
            # Map chunk IDs back to indices and content
            chunk_id_to_data = {
                chunk.chunk_id: (chunk.chunk_index, chunk.content) 
                for chunk in self.document_chunk_objects
            }
            
            for chunk_id, similarity_score in vector_results[:top_k]:
                if chunk_id in chunk_id_to_data:
                    chunk_index, chunk_content = chunk_id_to_data[chunk_id]
                    results.append((chunk_index, chunk_content, similarity_score))
            
            return results
        else:
            print(f"Falling back to keyword-based search")
            return self.keyword_based_search(question, top_k)
    
    def keyword_based_search(self, question: str, top_k: int = 3) -> List[Tuple[int, str, float]]:
        """Fallback keyword-based search when vector search is unavailable"""
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        chunk_scores = []
        
        for chunk in self.document_chunk_objects:
            chunk_lower = chunk.content.lower()
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            
            # Calculate relevance score
            score = 0.0
            
            # Keyword overlap (Jaccard similarity)
            if question_words and chunk_words:
                intersection = question_words.intersection(chunk_words)
                union = question_words.union(chunk_words)
                jaccard_score = len(intersection) / len(union) if union else 0
                score += jaccard_score * 3
            
            # Direct phrase matching
            question_phrases = [phrase.strip() for phrase in question_lower.split() if len(phrase.strip()) > 3]
            for phrase in question_phrases:
                if phrase in chunk_lower:
                    score += 2
            
            chunk_scores.append((chunk.chunk_index, chunk.content, score))
        
        # Sort by relevance score and return top_k
        chunk_scores.sort(key=lambda x: x[2], reverse=True)
        return chunk_scores[:top_k]
        
        # 1. Fix the missing extract_text_pypdf2 method
    def extract_text_pypdf2(self, pdf_path):
        """Extract text using PyPDF2 (faster but less accurate)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                page_count = len(pdf_reader.pages)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text, page_count
        except Exception as e:
            print(f"Error extracting text with PyPDF2: {e}")
            return None, 0

    def generate_answer(self, question: str, relevant_chunks: List[Tuple[int, str, float]]) -> QAResult:
        """
        Generate an answer based on relevant chunks
        
        Args:
            question (str): The question to answer
            relevant_chunks: List of relevant chunks with scores
            
        Returns:
            QAResult: The answer result
        """
        start_time = datetime.now()
        
        if not relevant_chunks:
            return QAResult(
                question=question,
                answer="I couldn't find relevant information in the document to answer this question.",
                confidence="low",
                source_chunks=[],
                processing_time=0.0
            )
        
        # Prepare context from relevant chunks
        context_parts = []
        source_chunk_indices = []
        
        for i, (chunk_idx, chunk_text, score) in enumerate(relevant_chunks):
            context_parts.append(f"[Context {i+1}]:\n{chunk_text}")
            source_chunk_indices.append(chunk_idx)
        
        context = "\n\n".join(context_parts)
        
        # Create a comprehensive prompt
        prompt = f"""Based on the following context from the document, please answer the question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
- Answer based ONLY on the information provided in the context
- If the context doesn't contain enough information, say so clearly
- Be specific and cite relevant details from the context
- If you're making any inferences, make that clear
- Keep your answer focused and relevant to the question

ANSWER:"""

        # Get answer from LLM
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Lower temperature for more focused answers
                "top_p": 0.9,
                "max_tokens": 1500
            }
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            answer = result.get('response', 'No answer generated')
            
            # Determine confidence based on relevance scores
            avg_score = sum(score for _, _, score in relevant_chunks) / len(relevant_chunks)
            if avg_score > 2.0:
                confidence = "high"
            elif avg_score > 1.0:
                confidence = "medium"
            else:
                confidence = "low"
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "Error occurred while generating answer"
            confidence = "low"
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return QAResult(
            question=question,
            answer=answer,
            confidence=confidence,
            source_chunks=source_chunk_indices,
            processing_time=processing_time
        )
    def ask_question(self, question: str, top_k: int = 3) -> QAResult:
        """
        Main method to ask a question about the loaded document with persistence
        
        Args:
            question (str): Question to ask
            top_k (int): Number of relevant chunks to consider
            
        Returns:
            QAResult: The answer result
        """
        if not self.document_chunk_objects:
            return QAResult(
                question=question,
                answer="No document has been loaded. Please summarize a PDF first to enable Q&A.",
                confidence="low",
                source_chunks=[],
                processing_time=0.0
            )
            
        print(f"🔍 Searching for relevant information to answer: '{question}'")
            
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(question, top_k)
            
        if relevant_chunks:
            print(f"Found {len(relevant_chunks)} relevant sections")
            for i, (chunk_idx, _, score) in enumerate(relevant_chunks):
                print(f"  Section {chunk_idx + 1}: relevance score {score:.2f}")
        else:
            print("No relevant sections found")
            return QAResult(
                question=question,
                answer="I couldn't find relevant information in the document to answer your question.",
                confidence="low",
                source_chunks=[],
                processing_time=0.0
            )
            
        # Generate answer
        result = self.generate_answer(question, relevant_chunks)
            
        # Save to Q&A history if we have a current session
        if self.current_session:
            self.vector_db.save_qa_result(self.current_session.session_id, result)
            
        return result    
    def interactive_qa_session(self):
        """
        Start an interactive Q&A session
        """
        if not self.document_chunks:
            print("No document loaded. Please summarize a PDF first.")
            return
        
        print(f"\nInteractive Q&A Session")
        print(f"Document: {Path(self.document_path).name if self.document_path else 'Unknown'}")
        print(f"Total sections: {len(self.document_chunks)}")
        print("Type 'quit', 'exit', or 'q' to end the session\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("Goodbye!")
                    break
                
                result = self.ask_question(question)
                
                print(f"\n**Answer** (confidence: {result.confidence}):")
                print(result.answer)
                print(f"\nBased on sections: {[i+1 for i in result.source_chunks]}")
                print(f"⏱Processing time: {result.processing_time:.2f}s")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def save_qa_session(self, qa_results: List[QAResult], output_path: str = None):
        """
        Save Q&A session results to a markdown file
        
        Args:
            qa_results: List of QAResult objects
            output_path: Path to save the file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qa_session_{timestamp}.md"
            output_path = Path("summaries") / filename
        
        Path(output_path).parent.mkdir(exist_ok=True)
        
        markdown_content = f"""# Q&A Session Results

**Document:** {Path(self.document_path).name if self.document_path else 'Unknown'}
**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model Used:** {self.model_name}
**Total Questions:** {len(qa_results)}

---

"""
        
        for i, result in enumerate(qa_results, 1):
            markdown_content += f"""## Question {i}

**Q:** {result.question}

**A:** {result.answer}

**Confidence:** {result.confidence.title()}
**Source Sections:** {[i+1 for i in result.source_chunks]}
**Processing Time:** {result.processing_time:.2f} seconds

---

"""
        
        markdown_content += "*This Q&A session was generated using Ollama with intelligent document chunking.*"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Q&A session saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error saving Q&A session: {e}")
            return None
    
    def split_by_sections(self, text: str) -> List[str]:
        """Split text by major sections/headings"""
        # Common section patterns
        section_patterns = [
            r'\n\s*(?:Chapter|CHAPTER|Section|SECTION)\s+\d+.*?\n',  # Chapter/Section headers
            r'\n\s*\d+\.\s+[A-Z][^\n]*\n',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]{10,}[A-Z]\s*\n',  # ALL CAPS headings
            r'\n\s*#{1,6}\s+.*?\n',  # Markdown headers
        ]
        
        # Try each pattern
        for pattern in section_patterns:
            sections = re.split(pattern, text)
            if len(sections) > 1:
                # Keep the headers with their content
                headers = re.findall(pattern, text)
                result = []
                for i, section in enumerate(sections):
                    if i > 0 and i-1 < len(headers):
                        result.append(headers[i-1] + section)
                    elif section.strip():
                        result.append(section)
                return result
        
        # No clear sections found, return as single piece
        return [text]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split large sections by paragraphs while respecting token limits"""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would exceed limit
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self.count_tokens(test_chunk) <= self.max_chunk_tokens:
                current_chunk = test_chunk
            else:
                # Current chunk is full, start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Check if single paragraph is too large
                if self.count_tokens(paragraph) > self.max_chunk_tokens:
                    # Split by sentences
                    sentence_chunks = self.split_by_sentences(paragraph)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences while respecting token limits"""
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) <= self.max_chunk_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence is still too large, split by words (last resort)
                if self.count_tokens(sentence) > self.max_chunk_tokens:
                    word_chunks = self.split_by_words(sentence)
                    chunks.extend(word_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split_by_words(self, text: str) -> List[str]:
        """Last resort: split by words (should rarely be needed)"""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            
            if self.count_tokens(test_chunk) <= self.max_chunk_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlapping content between chunks for context preservation"""
        if len(chunks) <= 1 or self.overlap_tokens <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            # Get overlap from previous chunk
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Extract last part of previous chunk for overlap
            prev_words = prev_chunk.split()
            overlap_text = ""
            
            # Build overlap by adding words from end of previous chunk
            for j in range(len(prev_words) - 1, -1, -1):
                test_overlap = " ".join(prev_words[j:]) + "\n\n" + current_chunk
                if self.count_tokens(test_overlap) <= self.max_chunk_tokens + self.overlap_tokens:
                    overlap_text = " ".join(prev_words[j:])
                else:
                    break
            
            # Create overlapped chunk
            if overlap_text:
                overlapped_chunk = f"[...continued from previous section...]\n{overlap_text}\n\n{current_chunk}"
            else:
                overlapped_chunk = current_chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def calculate_text_statistics(self, text: str, page_count: int = 0) -> DocumentStats:
        """
        Calculate comprehensive statistics for text
        
        Args:
            text (str): Input text
            page_count (int): Number of pages (for PDFs)
        
        Returns:
            DocumentStats: Statistics object
        """
        stats = DocumentStats()
        
        # Basic counts
        stats.pages = page_count
        stats.characters = len(text)
        stats.words = len(text.split()) if text.strip() else 0
        
        # Sentence count (simple approach - count sentence endings)
        sentence_endings = re.findall(r'[.!?]+', text)
        stats.sentences = len(sentence_endings)
        
        # Paragraph count (count double newlines or single newlines followed by whitespace)
        paragraphs = re.split(r'\n\s*\n|\n(?=\s)', text.strip())
        stats.paragraphs = len([p for p in paragraphs if p.strip()])
        
        return stats
        """Extract text using PyPDF2 (faster but less accurate)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text with PyPDF2: {e}")
            return None
    
    def extract_text_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (more accurate but slower)"""
        try:
            text = ""
            page_count = 0
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text, page_count
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
            return None, 0
    
    def extract_text_from_pdf(self, pdf_path, method="pdfplumber"):
        """
        Extract text from PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            method (str): Extraction method ('pypdf2' or 'pdfplumber')
        
        Returns:
            tuple: (extracted_text, page_count)
        """
        print(f"Extracting text from {pdf_path} using {method}...")
        
        if method == "pypdf2":
            text, page_count = self.extract_text_pypdf2(pdf_path)
        else:
            text, page_count = self.extract_text_pdfplumber(pdf_path)
        
        if text is None:
            # Fallback to the other method
            fallback_method = "pypdf2" if method == "pdfplumber" else "pdfplumber"
            print(f"Trying fallback method: {fallback_method}")
            if fallback_method == "pypdf2":
                text, page_count = self.extract_text_pypdf2(pdf_path)
            else:
                text, page_count = self.extract_text_pdfplumber(pdf_path)
        
        return (text if text else "", page_count)
    
    def chunk_text(self, text, max_chunk_size=4000):
        """
        Legacy method - replaced by intelligent_text_splitter
        Kept for backward compatibility
        """
        print("Warning: Using legacy chunking. Consider using intelligent_text_splitter() instead.")
        return self.intelligent_text_splitter(text)
    
    def get_summary_from_ollama(self, text, summary_type="comprehensive"):
        

        """
        Get summary from Ollama API
        
        Args:
            text (str): Text to summarize
            summary_type (str): Type of summary ('brief', 'comprehensive', 'bullet_points')
        
        Returns:
            str: Summary text
        """
        # Define different prompt templates
        prompts = {
            "brief": """Please provide a brief summary of the following text in 2-3 paragraphs:

{text}

Summary:""",
            "comprehensive": """Please provide a comprehensive summary of the following text. Include the main topics, key points, and important details:

{text}

Summary:""",
            "bullet_points": """Please summarize the following text using bullet points to highlight the main topics and key information:

{text}

Summary:"""
        }
        
        prompt = prompts.get(summary_type, prompts["comprehensive"]).format(text=text)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        try:
            print("Sending request to Ollama...")
            response = requests.post(self.api_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', 'No summary generated')
            
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing Ollama response: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Failed to connect to Ollama server: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during summarization: {e}")
            return None

    def save_summary_as_markdown(self, summary, pdf_path, stats: Union[SummaryStats, dict], output_dir="summaries"):
        """
        Save summary as markdown file with statistics
        
        Args:
            summary (str): Summary text
            pdf_path (str): Original PDF file path
            stats (Union[SummaryStats, dict]): Statistics object or dict
            output_dir (str): Output directory for markdown files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate markdown filename
        pdf_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = f"{pdf_name}_summary_{timestamp}.md"
        md_path = Path(output_dir) / md_filename
        
        # Handle both SummaryStats object and dict
        if isinstance(stats, dict):
            # Convert dict to SummaryStats if needed
            stats = SummaryStats(
                original=stats.get('original', {}),
                summary=stats.get('summary', {}),
                processing_time=stats.get('processing_time', 0)
            )
        
        # Create markdown content with statistics
        markdown_content = f"""# Summary of {Path(pdf_path).name}

    **Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    **Source File:** {pdf_path}
    **Model Used:** {self.model_name}

    {stats.get_summary_report()}

    ---

    ## Summary

    {summary}

    ---

    *This summary was generated automatically using Ollama with {self.model_name} model.*
    """
        
        # Save markdown file
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Summary saved to: {md_path}")
            return str(md_path)
        except Exception as e:
            print(f"Error saving markdown file: {e}")
            return None
        

    def summarize_pdf(self, pdf_path, summary_type="comprehensive", extraction_method="pdfplumber"):
        """
        Main method to summarize a PDF file with persistent session management
        
        Args:
            pdf_path (str): Path to the PDF file
            summary_type (str): Type of summary
            extraction_method (str): Text extraction method
        
        Returns:
            tuple: (path_to_summary_file, SummaryStats object, session_id)
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found: {pdf_path}")
            return None, None, None
        
        print(f"Processing PDF: {pdf_path}")
        start_time = datetime.now()
        
        # Load or create session
        session_id = self.load_or_create_session(pdf_path)
        
        # If session already exists with summary, ask user if they want to regenerate
        if (self.current_session and self.current_session.summary and 
            len(self.document_chunk_objects) > 0):
            
            print(f"Document already processed!")
            print(f"   - {len(self.document_chunk_objects)} chunks available")
            print(f"   - Created: {self.current_session.created_at}")
            print(f"   - Ready for Q&A")
            
            # Return existing summary info
            return None, None, session_id
        
        # Extract text from PDF (only if new session)
        text, page_count = self.extract_text_from_pdf(pdf_path, extraction_method)
        if not text.strip():
            print("Error: No text could be extracted from the PDF")
            return None, None, None
        
        # Calculate original document statistics
        original_stats = self.calculate_text_statistics(text, page_count)
        print(f"Original document: {original_stats}")
        
        # Store for session
        self.document_path = pdf_path
        self.document_stats = original_stats
        
        # Handle document chunking with intelligent splitting
        total_tokens = self.count_tokens(text)
        print(f"Document has {total_tokens:,} tokens")
        
        if total_tokens > self.max_chunk_tokens:
            print("Document is large, using intelligent chunking...")
            text_chunks = self.intelligent_text_splitter(text)
            print(f"Split into {len(text_chunks)} intelligent chunks")
        else:
            text_chunks = [text]
            print("Document fits in single chunk")
        
        # Create chunk objects with embeddings
        self.document_chunk_objects = self.create_chunk_objects(text_chunks)
        self.document_chunks = [chunk.content for chunk in self.document_chunk_objects]
        
        # Build vector index
        self.vector_db.build_faiss_index(self.document_chunk_objects)
        
        # Generate summaries
        summaries = []
        
        for i, chunk in enumerate(text_chunks, 1):
            chunk_tokens = self.count_tokens(chunk)
            print(f"Processing chunk {i}/{len(text_chunks)} ({chunk_tokens:,} tokens)")
            chunk_summary = self.get_summary_from_ollama(chunk, summary_type)
            if chunk_summary:
                summaries.append(f"## Section {i}\n\n{chunk_summary}")
        
        if not summaries:
            print("Error: Could not generate summary for any chunk")
            return None, None, session_id
        
        final_summary = "\n\n".join(summaries)
        
        # Get consolidated summary for multiple chunks
        if len(text_chunks) > 1:
            print("Creating consolidated summary...")
            consolidation_prompt = f"Please provide a consolidated summary that combines and synthesizes the key points from these section summaries:\n\n{final_summary}"
            consolidated = self.get_summary_from_ollama(consolidation_prompt, "comprehensive")
            if consolidated:
                final_summary = f"{consolidated}\n\n---\n\n# Detailed Section Summaries\n\n{final_summary}"
        
        # Calculate summary statistics
        summary_stats = self.calculate_text_statistics(final_summary)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create comprehensive stats object
        stats = SummaryStats(
            original=original_stats,
            summary=summary_stats,
            processing_time=processing_time
        )
        
        # Print statistics to console
        print("\n" + "="*50)
        print("SUMMARIZATION STATISTICS")
        print("="*50)
        print(f"Original: {original_stats}")
        print(f"Summary: {summary_stats}")
        stats.calculate_ratios()
        print(f"Compression: {stats.compression_ratio_words:.1f}% words, {stats.compression_ratio_chars:.1f}% characters")
        print(f"Processing time: {processing_time:.2f} seconds")
        print("="*50)
        
        # Save session with summary and stats
        self.save_current_session(final_summary, asdict(stats))
        
        # Save summary as markdown with statistics
        output_path = self.save_summary_as_markdown(final_summary, pdf_path, stats)
        
        print(f"Session persisted: {session_id}")
        print(f"Ready for Q&A! Ask questions anytime.")
        
        return output_path, stats, session_id
    
    def list_available_sessions(self):
        """List all available sessions"""
        sessions = self.vector_db.list_sessions()
        
        if not sessions:
            print("No sessions found")
            return
        
        print("\nAvailable Sessions:")
        print("-" * 80)
        
        for session in sessions:
            pdf_name = Path(session['pdf_path']).name
            created = datetime.fromisoformat(session['created_at']).strftime("%Y-%m-%d %H:%M")
            
            print(f"{pdf_name}")
            print(f"   ID: {session['session_id'][:16]}...")
            print(f"   Created: {created}")
            print(f"   Chunks: {session['total_chunks']}")
            print(f"   Model: {session['model_name']}")
            print()
    
    def load_session_by_id(self, session_id: str) -> bool:
        """Load a specific session by ID"""
        session, chunks = self.vector_db.load_session(session_id)
        
        if not session or not chunks:
            print(f"Session not found: {session_id}")
            return False
        
        self.current_session = session
        self.document_path = session.pdf_path
        self.document_chunk_objects = chunks
        self.document_chunks = [chunk.content for chunk in chunks]
        
        # Rebuild FAISS index
        self.vector_db.build_faiss_index(chunks)
        
        print(f"Loaded session: {Path(session.pdf_path).name}")
        print(f"   - {len(chunks)} chunks available")
        print(f"   - Created: {session.created_at}")
        print(f"   - Ready for Q&A")
        
        return True
    
    def show_qa_history(self, limit: int = 10):
        """Show Q&A history for current session"""
        if not self.current_session:
            print("No active session")
            return
        
        history = self.vector_db.get_qa_history(self.current_session.session_id, limit)
        
        if not history:
            print("No Q&A history found")
            return
        
        print(f"\nQ&A History (last {len(history)} questions):")
        print("-" * 80)
        
        for i, qa in enumerate(reversed(history), 1):
            print(f"Q{i}: {qa.question}")
            print(f"A{i}: {qa.answer[:200]}{'...' if len(qa.answer) > 200 else ''}")
            print(f"   Confidence: {qa.confidence} | Time: {qa.processing_time:.2f}s")
            print()
    
    def delete_session(self, session_id: str = None):
        """Delete a session"""
        target_id = session_id or (self.current_session.session_id if self.current_session else None)
        
        if not target_id:
            print("No session to delete")
            return
        
        self.vector_db.delete_session(target_id)
        
        if target_id == (self.current_session.session_id if self.current_session else None):
            self.current_session = None
            self.document_chunks = []
            self.document_chunk_objects = []
            self.document_path = None
        
        print(f"Deleted session: {target_id}")
    
    def enhanced_interactive_qa(self):
        """Enhanced interactive Q&A with session management commands"""
        if not self.document_chunk_objects:
            print("No document loaded. Please process a PDF first.")
            return
        
        print(f"\nEnhanced Interactive Q&A Session")
        print(f"Document: {Path(self.document_path).name if self.document_path else 'Unknown'}")
        print(f"Session ID: {self.current_session.session_id[:16]}... " if self.current_session else "No session")
        print(f"Total sections: {len(self.document_chunk_objects)}")
        print("\nCommands:")
        print("  - Ask any question about the document")
        print("  - 'history' or 'h' - Show Q&A history")
        print("  - 'stats' or 's' - Show document statistics")
        print("  - 'quit', 'exit', or 'q' - End session")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n❓ Your question or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() in ['history', 'h']:
                    self.show_qa_history()
                    continue
                elif user_input.lower() in ['stats', 's']:
                    if self.document_stats:
                        print(f"\nDocument Statistics:")
                        print(f"   {self.document_stats}")
                    continue
                elif not user_input:
                    continue
                
                # Process question
                result = self.ask_question(user_input)
                
                print(f"\n**Answer** (confidence: {result.confidence}):")
                print(result.answer)
                print(f"\nBased on sections: {[i+1 for i in result.source_chunks]}")
                print(f"⏱Processing time: {result.processing_time:.2f}s")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
        """
        Main method to summarize a PDF file with statistics tracking
        
        Args:
            pdf_path (str): Path to the PDF file
            summary_type (str): Type of summary
            extraction_method (str): Text extraction method
        
        Returns:
            tuple: (path_to_summary_file, SummaryStats object)
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found: {pdf_path}")
            return None, None
        
        print(f"Starting PDF summarization for: {pdf_path}")
        start_time = datetime.now()
        
        # Extract text from PDF
        text, page_count = self.extract_text_from_pdf(pdf_path, extraction_method)
        if not text.strip():
            print("Error: No text could be extracted from the PDF")
            return None, None
        
        # Calculate original document statistics
        original_stats = self.calculate_text_statistics(text, page_count)
        print(f"Original document: {original_stats}")
        
        # Store chunks for Q&A functionality
        self.document_path = pdf_path
        self.document_stats = original_stats
        
        # Handle large documents with intelligent chunking
        total_tokens = self.count_tokens(text)
        print(f"Document has {total_tokens:,} tokens")
        
        if total_tokens > self.max_chunk_tokens:
            print("Document is large, using intelligent chunking...")
            chunks = self.intelligent_text_splitter(text)
            print(f"Split into {len(chunks)} intelligent chunks")
            
            # Store chunks for Q&A
            self.document_chunks = chunks
            
            summaries = []
            
            for i, chunk in enumerate(chunks, 1):
                chunk_tokens = self.count_tokens(chunk)
                print(f"Processing chunk {i}/{len(chunks)} ({chunk_tokens:,} tokens)")
                chunk_summary = self.get_summary_from_ollama(chunk, summary_type)
                if chunk_summary:
                    summaries.append(f"## Section {i}\n\n{chunk_summary}")
            
            if summaries:
                final_summary = "\n\n".join(summaries)
                # Get a final consolidated summary if there are multiple chunks
                if len(chunks) > 1:
                    print("Creating consolidated summary...")
                    consolidation_prompt = f"Please provide a consolidated summary that combines and synthesizes the key points from these section summaries:\n\n{final_summary}"
                    consolidated = self.get_summary_from_ollama(consolidation_prompt, "comprehensive")
                    if consolidated:
                        final_summary = f"{consolidated}\n\n---\n\n# Detailed Section Summaries\n\n{final_summary}"
            else:
                print("Error: Could not generate summary for any chunk")
                return None, None
        else:
            # Process entire document at once
            self.document_chunks = [text]  # Store single chunk for Q&A
            final_summary = self.get_summary_from_ollama(text, summary_type)
            if not final_summary:
                print("Error: Could not generate summary")
                return None, None
        
        # Calculate summary statistics
        summary_stats = self.calculate_text_statistics(final_summary)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create comprehensive stats object
        stats = SummaryStats(
            original=original_stats,
            summary=summary_stats,
            processing_time=processing_time
        )
        
        # Print statistics to console
        print("\n" + "="*50)
        print("SUMMARIZATION STATISTICS")
        print("="*50)
        print(f"Original: {original_stats}")
        print(f"Summary: {summary_stats}")
        stats.calculate_ratios()
        print(f"Compression: {stats.compression_ratio_words:.1f}% words, {stats.compression_ratio_chars:.1f}% characters")
        print(f"Processing time: {processing_time:.2f} seconds")
        print("="*50)
        
        # Save summary as markdown with statistics
        output_path = self.save_summary_as_markdown(final_summary, pdf_path, stats)
        return output_path, stats

def main():
    parser = argparse.ArgumentParser(description="PDF Summarizer with Persistent Vector Database")
    
    # Document processing arguments
    parser.add_argument("pdf_path", nargs='?', help="Path to the PDF file to summarize")
    parser.add_argument("--model", default="mistral", help="Ollama model name (default: mistral)")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--type", choices=["brief", "comprehensive", "bullet_points"], 
                       default="comprehensive", help="Type of summary")
    parser.add_argument("--method", choices=["pypdf2", "pdfplumber"], 
                       default="pdfplumber", help="Text extraction method")
    
    # Chunking configuration
    parser.add_argument("--max-tokens", type=int, default=3000, help="Maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap tokens between chunks")
    
    # Database configuration
    parser.add_argument("--db", default="pdf_analyzer.db", help="Vector database path")
    
    # Q&A options
    parser.add_argument("--qa", action="store_true", help="Start interactive Q&A after processing")
    parser.add_argument("--question", type=str, help="Ask a single question")
    parser.add_argument("--qa-only", action="store_true", help="Skip processing, only Q&A")
    
    # Session management
    parser.add_argument("--load-session", type=str, help="Load specific session by ID")
    parser.add_argument("--list-sessions", action="store_true", help="List all available sessions")
    parser.add_argument("--delete-session", type=str, help="Delete specific session by ID")
    parser.add_argument("--history", action="store_true", help="Show Q&A history for current session")
    
    # Output
    parser.add_argument("--output", default="summaries", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize summarizer with vector database
    summarizer = PDFSummarizer(
        ollama_url=args.url,
        model_name=args.model,
        max_chunk_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        db_path=args.db
    )
    
    # Session management commands
    if args.list_sessions:
        summarizer.list_available_sessions()
        return
    
    if args.delete_session:
        summarizer.delete_session(args.delete_session)
        return
    
    if args.load_session:
        if summarizer.load_session_by_id(args.load_session):
            if args.question:
                # Single question mode
                result = summarizer.ask_question(args.question)
                print(f"\n**Question:** {result.question}")
                print(f"**Answer:** {result.answer}")
                print(f"**Confidence:** {result.confidence}")
                return
            else:
                # Interactive mode
                summarizer.enhanced_interactive_qa()
        return
    
    if args.history:
        if args.pdf_path:
            session_id = summarizer.load_or_create_session(args.pdf_path)
            summarizer.show_qa_history()
        else:
            print("PDF path required to show history")
        return
    
    # Q&A only mode (load existing session for PDF)
    if args.qa_only:
        if not args.pdf_path:
            print("PDF path required for Q&A mode")
            return
        
        session_id = summarizer.vector_db.session_exists(args.pdf_path)
        if not session_id:
            print("No existing session found. Please process the PDF first.")
            print("   Run without --qa-only to process the document.")
            return
        
        if summarizer.load_session_by_id(session_id):
            if args.question:
                result = summarizer.ask_question(args.question)
                print(f"\n**Question:** {result.question}")
                print(f"**Answer:** {result.answer}")
                print(f"**Confidence:** {result.confidence}")
            else:
                summarizer.enhanced_interactive_qa()
        return
    
    # Main processing mode
    if not args.pdf_path:
        print("PDF path is required")
        parser.print_help()
        return
    
    # Process PDF (will load existing session if available)
    result, stats, session_id = summarizer.summarize_pdf(
        pdf_path=args.pdf_path,
        summary_type=args.type,
        extraction_method=args.method
    )
    
    if session_id:
        if result:  # New processing completed
            print(f"\nSummary saved to: {result}")
            if stats:
                print(f"\nQuick Stats:")
                print(f"  Original: {stats.original.pages} pages, {stats.original.words:,} words")
                print(f"  Summary: {stats.summary.words:,} words ({stats.compression_ratio_words:.1f}% of original)")
                print(f"  Processing time: {stats.processing_time:.2f} seconds")
        
        print(f"\nSession ID: {session_id}")
        print("Tip: Use --qa-only next time to skip reprocessing")
        
        # Handle Q&A options
        if args.question:
            print(f"\nAnswering your question...")
            qa_result = summarizer.ask_question(args.question)
            print(f"\n**Question:** {qa_result.question}")
            print(f"**Answer:** {qa_result.answer}")
            print(f"**Confidence:** {qa_result.confidence}")
        elif args.qa:
            summarizer.enhanced_interactive_qa()
        else:
            print(f"\nReady for questions! Use:")
            print(f"   python {sys.argv[0]} \"{args.pdf_path}\" --qa-only --question \"Your question\"")
            print(f"   python {sys.argv[0]} \"{args.pdf_path}\" --qa-only  # Interactive mode")
    else:
        print("❌ Failed to process document")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Summarize PDF files using Ollama with Q&A capabilities")
    parser.add_argument("pdf_path", nargs='?', help="Path to the PDF file to summarize")
    parser.add_argument("--model", default="mistral", help="Ollama model name (default: mistral)")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--type", choices=["brief", "comprehensive", "bullet_points"], 
                       default="comprehensive", help="Type of summary (default: comprehensive)")
    parser.add_argument("--method", choices=["pypdf2", "pdfplumber"], 
                       default="pdfplumber", help="Text extraction method (default: pdfplumber)")
    parser.add_argument("--max-tokens", type=int, default=3000, 
                       help="Maximum tokens per chunk (default: 3000)")
    parser.add_argument("--overlap", type=int, default=200, 
                       help="Overlap tokens between chunks (default: 200)")
    parser.add_argument("--output", default="summaries", help="Output directory for summaries (default: summaries)")
    parser.add_argument("--qa", action="store_true", help="Start interactive Q&A session after summarization")
    parser.add_argument("--question", type=str, help="Ask a single question about the document")
    parser.add_argument("--qa-only", action="store_true", help="Skip summarization, only do Q&A (requires previous processing)")
    
    args = parser.parse_args()
    
    # Initialize summarizer with configurable chunking parameters
    summarizer = PDFSummarizer(
        ollama_url=args.url, 
        model_name=args.model,
        max_chunk_tokens=args.max_tokens,
        overlap_tokens=args.overlap
    )
    
    # Handle Q&A only mode
    if args.qa_only:
        if not args.pdf_path:
            print("PDF path required for Q&A mode")
            sys.exit(1)
        
        print("Loading document for Q&A...")
        # Load document without summarization
        if not os.path.exists(args.pdf_path):
            print(f"Error: PDF file not found: {args.pdf_path}")
            sys.exit(1)
        
        text, page_count = summarizer.extract_text_from_pdf(args.pdf_path, args.method)
        if not text.strip():
            print("Error: No text could be extracted from the PDF")
            sys.exit(1)
        
        summarizer.document_path = args.pdf_path
        summarizer.document_stats = summarizer.calculate_text_statistics(text, page_count)
        
        total_tokens = summarizer.count_tokens(text)
        if total_tokens > summarizer.max_chunk_tokens:
            chunks = summarizer.intelligent_text_splitter(text)
            summarizer.document_chunks = chunks
        else:
            summarizer.document_chunks = [text]
        
        print(f"Document loaded: {len(summarizer.document_chunks)} sections")
        
        if args.question:
            # Single question mode
            result = summarizer.ask_question(args.question)
            print(f"\n**Question:** {result.question}")
            print(f"**Answer:** {result.answer}")
            print(f"**Confidence:** {result.confidence}")
            print(f"⏱**Processing time:** {result.processing_time:.2f}s")
        else:
            # Interactive mode
            summarizer.interactive_qa_session()
        
        return
    
    # Regular summarization mode
    if not args.pdf_path:
        print("PDF path is required")
        parser.print_help()
        sys.exit(1)
    
    # Summarize PDF
    result, stats, session_id = summarizer.summarize_pdf(
        pdf_path=args.pdf_path,
        summary_type=args.type,
        extraction_method=args.method
    )
    
    if result:
        print(f"\nSuccess! Summary saved to: {result}")
        if stats:
            print(f"\nQuick Stats:")
            print(f"  Original: {stats.original.pages} pages, {stats.original.words:,} words")
            print(f"  Summary: {stats.summary.words:,} words ({stats.compression_ratio_words:.1f}% of original)")
            print(f"  Processing time: {stats.processing_time:.2f} seconds")
        
        # Handle Q&A options
        if args.question:
            print(f"\nAnswering your question...")
            qa_result = summarizer.ask_question(args.question)
            print(f"\n**Question:** {qa_result.question}")
            print(f"**Answer:** {qa_result.answer}")
            print(f"**Confidence:** {qa_result.confidence}")
            
        elif args.qa:
            print(f"\nStarting interactive Q&A session...")
            summarizer.interactive_qa_session()
    else:
        print("\nFailed to generate summary")
        sys.exit(1)

if __name__ == "__main__":
    main()
