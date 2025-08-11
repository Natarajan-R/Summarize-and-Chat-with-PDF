import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import threading
import time
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
import uuid

# Import your existing PDFSummarizer class
from pdf_summarizer import PDFSummarizer, QAResult, SummaryStats


def dataclass_to_dict(obj):
    """Convert dataclass objects to dictionaries for JSON serialization"""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    else:
        return obj
    

# Configure logging
def setup_logging():
    """Setup comprehensive logging"""
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('logs/app.log', maxBytes=10485760, backupCount=5),  # 10MB
            logging.StreamHandler()  # Console output
        ]
    )
    
    # Create specific loggers
    app_logger = logging.getLogger('PDFAnalyzerApp')
    socketio_logger = logging.getLogger('SocketIO')
    processing_logger = logging.getLogger('Processing')
    
    return app_logger, socketio_logger, processing_logger

def format_stats_for_frontend(stats):
    """Format statistics for frontend display"""
    try:
        if not stats:
            app_logger.warning("No stats provided to format_stats_for_frontend")
            return None
        
        app_logger.info(f"Raw stats object type: {type(stats)}")
        app_logger.info(f"Raw stats object: {stats}")
        
        # Handle both SummaryStats object and dict formats
        if hasattr(stats, '__dict__'):
            # It's a SummaryStats object
            app_logger.info("Processing SummaryStats object")
            
            original_stats = getattr(stats, 'original', None)
            summary_stats = getattr(stats, 'summary', None)
            
            app_logger.info(f"Original stats: {original_stats}")
            app_logger.info(f"Summary stats: {summary_stats}")
            
            # Use the correct attribute names from DocumentStats
            formatted_stats = {
                'original': {
                    'page_count': getattr(original_stats, 'pages', 0) if original_stats else 0,
                    'word_count': getattr(original_stats, 'words', 0) if original_stats else 0,
                    'char_count': getattr(original_stats, 'characters', 0) if original_stats else 0,
                    'reading_time': calculate_reading_time(getattr(original_stats, 'words', 0)) if original_stats else 0
                },
                'summary': {
                    'page_count': getattr(summary_stats, 'pages', 0) if summary_stats else 0,
                    'word_count': getattr(summary_stats, 'words', 0) if summary_stats else 0,
                    'char_count': getattr(summary_stats, 'characters', 0) if summary_stats else 0,
                    'reading_time': calculate_reading_time(getattr(summary_stats, 'words', 0)) if summary_stats else 0
                },
                'processing_time': getattr(stats, 'processing_time', 0)
            }
            
        else:
            # It's already a dict - handle both possible formats
            app_logger.info("Processing dictionary format")
            
            # Check if it's using the old format (page_count, word_count, etc.)
            original_dict = stats.get('original', {})
            summary_dict = stats.get('summary', {})
            
            formatted_stats = {
                'original': {
                    'page_count': original_dict.get('page_count') or original_dict.get('pages', 0),
                    'word_count': original_dict.get('word_count') or original_dict.get('words', 0),
                    'char_count': original_dict.get('char_count') or original_dict.get('characters', 0),
                    'reading_time': original_dict.get('reading_time', 0) or calculate_reading_time(
                        original_dict.get('word_count') or original_dict.get('words', 0)
                    )
                },
                'summary': {
                    'page_count': summary_dict.get('page_count') or summary_dict.get('pages', 0),
                    'word_count': summary_dict.get('word_count') or summary_dict.get('words', 0),
                    'char_count': summary_dict.get('char_count') or summary_dict.get('characters', 0),
                    'reading_time': summary_dict.get('reading_time', 0) or calculate_reading_time(
                        summary_dict.get('word_count') or summary_dict.get('words', 0)
                    )
                },
                'processing_time': stats.get('processing_time', 0)
            }
            
        app_logger.info(f"Formatted stats for frontend: {formatted_stats}")
        return formatted_stats
        
    except Exception as e:
        app_logger.error(f"Error formatting stats for frontend: {str(e)}", exc_info=True)
        return None


def calculate_reading_time(word_count):
    """Calculate estimated reading time in minutes (assuming 200 words per minute)"""
    if not word_count or word_count <= 0:
        return 0
    return round(word_count / 200, 1)  # 200 words per minute average reading speed


app_logger, socketio_logger, processing_logger = setup_logging()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize SocketIO with Python 3.12 compatible settings
# Use 'threading' async_mode instead of 'eventlet' to avoid SSL issues
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    logger=True, 
    engineio_logger=True,
    async_mode='threading',  # Use threading instead of eventlet
    ping_timeout=60,
    ping_interval=25
)

# Global storage for active sessions
active_sessions = {}
processing_status = {}
active_tasks = {}  # Track active processing tasks

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
app_logger.info(f"Upload directory created/verified: {app.config['UPLOAD_FOLDER']}")

class WebPDFSummarizer(PDFSummarizer):
    """Extended PDFSummarizer with web-specific features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = None
        self.task_id = None
    
    def set_progress_callback(self, callback, task_id=None):
        """Set callback for progress updates"""
        self.progress_callback = callback
        self.task_id = task_id
    
    def emit_progress(self, message: str, progress: int = None):
        """Emit progress updates via callback"""
        processing_logger.info(f"Task {self.task_id}: {message} (Progress: {progress}%)")
        if self.progress_callback:
            self.progress_callback(message, progress)

    def debug_stats_object(self, stats):
        """Debug the stats object structure"""
        try:
            processing_logger.info(f"=== STATS DEBUG ===")
            processing_logger.info(f"Stats type: {type(stats)}")
            processing_logger.info(f"Stats value: {stats}")
            
            if hasattr(stats, '__dict__'):
                processing_logger.info(f"Stats __dict__: {stats.__dict__}")
                processing_logger.info(f"Stats dir(): {[attr for attr in dir(stats) if not attr.startswith('_')]}")
                
                # Check each attribute
                for attr in ['original', 'summary', 'processing_time']:
                    if hasattr(stats, attr):
                        val = getattr(stats, attr)
                        processing_logger.info(f"stats.{attr} = {val} (type: {type(val)})")
                        if hasattr(val, '__dict__'):
                            processing_logger.info(f"stats.{attr}.__dict__ = {val.__dict__}")
                    else:
                        processing_logger.info(f"stats.{attr} = NOT FOUND")
            
            processing_logger.info(f"=== END STATS DEBUG ===")
        except Exception as e:
            processing_logger.error(f"Error in debug_stats_object: {e}")

    def summarize_pdf_with_progress(self, pdf_path, summary_type="comprehensive", extraction_method="pdfplumber"):
        """Enhanced summarize_pdf with progress updates"""
        processing_logger.info(f"Starting PDF summarization: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            processing_logger.error(error_msg)
            self.emit_progress(f"Error: {error_msg}")
            return None, None, None
        
        self.emit_progress(f"Processing PDF: {Path(pdf_path).name}", 10)
        start_time = datetime.now()
        
        try:
            # Load or create session
            session_id = self.load_or_create_session(pdf_path)
            self.emit_progress("Loading/creating session...", 20)
            processing_logger.info(f"Session ID: {session_id}")
            
            # Check if already processed
            if (self.current_session and self.current_session.summary and 
                len(self.document_chunk_objects) > 0):
                
                self.emit_progress("Document already processed!", 100)
                processing_logger.info("Document was already processed")
                return None, None, session_id
            
            # Extract text
            self.emit_progress("Extracting text from PDF...", 30)
            text, page_count = self.extract_text_from_pdf(pdf_path, extraction_method)
            
            if not text.strip():
                error_msg = "No text could be extracted from the PDF"
                processing_logger.error(error_msg)
                self.emit_progress(f"Error: {error_msg}")
                return None, None, None
            
            # Calculate stats
            original_stats = self.calculate_text_statistics(text, page_count)
            self.emit_progress(f"Analyzed document: {original_stats}", 40)
            processing_logger.info(f"Document stats: {original_stats}")
            
            self.document_path = pdf_path
            self.document_stats = original_stats
            
            # Handle chunking
            total_tokens = self.count_tokens(text)
            self.emit_progress(f"Document has {total_tokens:,} tokens", 50)
            processing_logger.info(f"Total tokens: {total_tokens}")
            
            if total_tokens > self.max_chunk_tokens:
                self.emit_progress("Creating intelligent chunks...", 60)
                text_chunks = self.intelligent_text_splitter(text)
                self.emit_progress(f"Split into {len(text_chunks)} chunks", 65)
                processing_logger.info(f"Split into {len(text_chunks)} chunks")
            else:
                text_chunks = [text]
                self.emit_progress("Document fits in single chunk", 65)
                processing_logger.info("Document fits in single chunk")
            
            # Create embeddings
            self.emit_progress("Creating embeddings...", 70)
            self.document_chunk_objects = self.create_chunk_objects(text_chunks)
            self.document_chunks = [chunk.content for chunk in self.document_chunk_objects]
            
            # Build vector index
            self.emit_progress("Building search index...", 75)
            self.vector_db.build_faiss_index(self.document_chunk_objects)
            
            # Generate summaries
            self.emit_progress("Generating summaries...", 80)
            summaries = []
            
            for i, chunk in enumerate(text_chunks, 1):
                chunk_tokens = self.count_tokens(chunk)
                progress_val = 80 + (10 * i / len(text_chunks))
                self.emit_progress(f"Processing section {i}/{len(text_chunks)} ({chunk_tokens:,} tokens)", 
                                 int(progress_val))
                
                chunk_summary = self.get_summary_from_ollama(chunk, summary_type)
                if chunk_summary:
                    summaries.append(f"## Section {i}\n\n{chunk_summary}")
                    processing_logger.info(f"Completed section {i}/{len(text_chunks)}")
            
            if not summaries:
                error_msg = "Could not generate summary"
                processing_logger.error(error_msg)
                self.emit_progress(f"Error: {error_msg}")
                return None, None, session_id
            
            final_summary = "\n\n".join(summaries)
            
            # Consolidate if multiple chunks
            if len(text_chunks) > 1:
                self.emit_progress("Creating consolidated summary...", 95)
                consolidation_prompt = f"Please provide a consolidated summary that combines and synthesizes the key points from these section summaries:\n\n{final_summary}"
                consolidated = self.get_summary_from_ollama(consolidation_prompt, "comprehensive")
                if consolidated:
                    final_summary = f"{consolidated}\n\n---\n\n# Detailed Section Summaries\n\n{final_summary}"
                        
            # Calculate final stats
            summary_stats = self.calculate_text_statistics(final_summary)
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            processing_logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            # Create stats object
            processing_logger.info(f"=== CREATING STATS OBJECT ===")
            processing_logger.info(f"original_stats: {original_stats}")
            processing_logger.info(f"summary_stats: {summary_stats}")
            processing_logger.info(f"processing_time: {processing_time}")
                            
            stats = SummaryStats(
                original=original_stats,
                summary=summary_stats,
                processing_time=processing_time
            )
            
            # Convert to dict for saving
            stats_dict = {
                'original': original_stats.__dict__,
                'summary': summary_stats.__dict__,
                'processing_time': processing_time
            }
            
            processing_logger.info(f"Created SummaryStats object: {stats}")
            processing_logger.info(f"SummaryStats object type: {type(stats)}")
            processing_logger.info(f"SummaryStats object __dict__: {stats.__dict__}")
            self.debug_stats_object(stats)
            processing_logger.info(f"=== END CREATING STATS OBJECT ===")
            
            # Save session with dict format
            self.save_current_session(final_summary, stats_dict)
            self.emit_progress("Session saved successfully", 98)
            
            # Save summary using the SummaryStats object
            output_path = self.save_summary_as_markdown(final_summary, pdf_path, stats)
            processing_logger.info(f"Summary saved to: {output_path}")
            
            self.emit_progress("Processing complete!", 100)
            
            return output_path, stats, session_id
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            processing_logger.error(error_msg, exc_info=True)
            self.emit_progress(f"{error_msg}")
            raise


def get_summarizer(session_id=None) -> WebPDFSummarizer:
    """Get or create a summarizer instance"""
    # Try to get session ID from parameter first, then from Flask session
    current_session_id = session_id
    if not current_session_id:
        try:
            current_session_id = session.get('session_id')
        except RuntimeError:
            # Working outside request context - create a new session ID
            current_session_id = str(uuid.uuid4())
            app_logger.info(f"Created session ID outside request context: {current_session_id}")
    
    if not current_session_id:
        current_session_id = str(uuid.uuid4())
        try:
            session['session_id'] = current_session_id
        except RuntimeError:
            # Can't set session outside request context, but that's okay
            pass
        app_logger.info(f"Created new session: {current_session_id}")
    
    if current_session_id not in active_sessions:
        app_logger.info(f"Creating new summarizer for session: {current_session_id}")
        active_sessions[current_session_id] = WebPDFSummarizer(
            model_name="mistral",  # Default model
            db_path="web_pdf_analyzer.db"
        )
    
    return active_sessions[current_session_id]


# Routes
@app.route('/')
def index():
    """Main dashboard"""
    app_logger.info("Serving main dashboard")
    try:
        summarizer = get_summarizer()
        
        # Try to get sessions, but handle gracefully if it fails
        sessions = []
        try:
            if hasattr(summarizer.vector_db, 'list_sessions'):
                sessions = summarizer.vector_db.list_sessions()
                app_logger.info(f"Found {len(sessions)} existing sessions")
            else:
                app_logger.warning("VectorDatabase doesn't have list_sessions method")
                sessions = []
        except Exception as sessions_error:
            app_logger.error(f"Error getting sessions for dashboard: {str(sessions_error)}")
            sessions = []
            
        return render_template('index.html', sessions=sessions)
    except Exception as e:
        app_logger.error(f"Error loading dashboard: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to load dashboard'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    app_logger.info("File upload request received")
    
    if 'file' not in request.files:
        app_logger.warning("No file in upload request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        app_logger.warning("Empty filename in upload request")
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        app_logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        app_logger.info(f"File uploaded successfully: {filename} ({file_size:,} bytes)")
        
        return jsonify({
            'success': True, 
            'filepath': filepath,
            'filename': file.filename,
            'size': file_size
        })
    except Exception as e:
        app_logger.error(f"Upload failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
        

@app.route('/process', methods=['POST'])
def process_pdf():
    """Start PDF processing"""
    data = request.json
    filepath = data.get('filepath')
    summary_type = data.get('summary_type', 'comprehensive')
    
    app_logger.info(f"Processing request: {filepath} (type: {summary_type})")
    
    if not filepath or not os.path.exists(filepath):
        app_logger.error(f"Invalid file path: {filepath}")
        return jsonify({'error': 'Invalid file path'}), 400
    
    # Get current session ID before starting background task
    current_session_id = session.get('session_id')
    if not current_session_id:
        current_session_id = str(uuid.uuid4())
        session['session_id'] = current_session_id
    
    # Create processing task
    task_id = str(uuid.uuid4())
    processing_status[task_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Initializing...',
        'result': None,
        'task_id': task_id,
        'created_at': datetime.now().isoformat()
    }
    
    active_tasks[task_id] = True
    app_logger.info(f"Created processing task: {task_id} for session: {current_session_id}")
    
    # Start processing in background with app context
    def process_task():
        processing_logger.info(f"Background task started: {task_id}")
        
        # Use application context for background thread
        with app.app_context():
            try:
                # Create summarizer instance directly without using Flask session
                if current_session_id not in active_sessions:
                    app_logger.info(f"Creating new summarizer for background task: {current_session_id}")
                    active_sessions[current_session_id] = WebPDFSummarizer(
                        model_name="mistral",
                        db_path="web_pdf_analyzer.db"
                    )
                
                summarizer = active_sessions[current_session_id]
                
                # Set up progress callback
                def progress_callback(message, progress=None):
                    if task_id not in active_tasks:
                        processing_logger.info(f"Task {task_id} cancelled")
                        return
                        
                    processing_status[task_id].update({
                        'message': message,
                        'progress': progress or processing_status[task_id]['progress'],
                        'status': 'processing',
                        'task_id': task_id
                    })
    
                    # Emit to SocketIO
                    socketio_logger.info(f"Emitting progress update for task {task_id}: {message}")
                    socketio.emit('progress_update', {
                        'task_id': task_id,
                        'progress': progress,
                        'message': message,
                        'status': 'processing'
                    }, room=f"task_{task_id}")
                
                summarizer.set_progress_callback(progress_callback, task_id)
                
                # Process PDF
                processing_logger.info(f"Starting PDF processing for task {task_id}")
                result_path, stats, pdf_session_id = summarizer.summarize_pdf_with_progress(
                    filepath, summary_type
                )
                
                # Update final status
                formatted_stats = format_stats_for_frontend(stats)
                
                summarizer.debug_stats_object(stats) 

                processing_status[task_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Processing completed successfully!',
                    'task_id': task_id,
                    'result': {
                        'session_id': pdf_session_id or current_session_id,
                        'result_path': result_path,
                        'stats': formatted_stats,
                        'document_path': summarizer.document_path,
                        'chunks_count': len(summarizer.document_chunk_objects)
                    },
                    'completed_at': datetime.now().isoformat()
                })

                processing_logger.info(f"Task {task_id} completed successfully with formatted stats: {formatted_stats}")
                
            except Exception as e:
                processing_logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
                processing_status[task_id].update({
                    'status': 'error',
                    'message': f'Error: {str(e)}',
                    'task_id': task_id,
                    'error_at': datetime.now().isoformat()
                })
            finally:
                # Clean up
                if task_id in active_tasks:
                    del active_tasks[task_id]
            
            # Emit completion event
            try:
                # Convert stats to dict format for JSON serialization
                completion_data = dataclass_to_dict(processing_status[task_id])
                socketio.emit('processing_complete', completion_data, room=f"task_{task_id}")
                processing_logger.info(f"Emitted processing_complete for task {task_id}")
            except Exception as e:
                processing_logger.error(f"Failed to emit processing_complete: {str(e)}")
    
    # Use threading instead of eventlet
    thread = threading.Thread(target=process_task, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'Processing started'
    })

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Get processing task status"""
    status = processing_status.get(task_id, {'status': 'not_found'})
    app_logger.info(f"Task status request for {task_id}: {status.get('status')}")
    return jsonify(status)

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel processing task"""
    if task_id in active_tasks:
        del active_tasks[task_id]
        processing_status[task_id].update({
            'status': 'cancelled',
            'message': 'Task cancelled by user',
            'task_id': task_id,
            'cancelled_at': datetime.now().isoformat()
        })
        app_logger.info(f"Task cancelled: {task_id}")
        return jsonify({'success': True, 'message': 'Task cancelled'})
    else:
        return jsonify({'error': 'Task not found or already completed'}), 404

@app.route('/qa', methods=['POST'])
def ask_question():
    """Handle Q&A requests"""
    data = request.json
    question = data.get('question', '').strip()
    session_id = data.get('session_id')
    
    app_logger.info(f"Q&A request: '{question}' for session {session_id}")
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    try:
        summarizer = get_summarizer()
        
        # Load session if specified
        if session_id and session_id != (summarizer.current_session.session_id if summarizer.current_session else None):    
            if not summarizer.load_session_by_id(session_id):
                app_logger.error(f"Session not found: {session_id}")
                return jsonify({'error': 'Session not found'}), 404
        
        # Ask question
        result = summarizer.ask_question(question)
        app_logger.info(f"Q&A completed with confidence: {result.confidence}")
        
        return jsonify({
            'question': result.question,
            'answer': result.answer,
            'confidence': result.confidence,
            'source_chunks': result.source_chunks,
            'processing_time': result.processing_time
        })
        
    except Exception as e:
        app_logger.error(f"Q&A failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Q&A failed: {str(e)}'}), 500

@app.route('/sessions')
def list_sessions():
    """Get all sessions"""
    try:
        summarizer = get_summarizer()
        
        # Try to get sessions, but handle if method fails
        sessions = []
        try:
            if hasattr(summarizer.vector_db, 'list_sessions'):
                sessions = summarizer.vector_db.list_sessions()
                app_logger.info(f"Listing {len(sessions)} sessions")
            else:
                app_logger.warning("VectorDatabase doesn't have list_sessions method")
                sessions = []
        except Exception as sessions_error:
            app_logger.error(f"Error listing sessions: {str(sessions_error)}")
            sessions = []
            
        return jsonify(sessions)
    except Exception as e:
        app_logger.error(f"Failed to list sessions: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to list sessions'}), 500

@app.route('/session/<session_id>')
def load_session(session_id):
    """Load specific session with enhanced information"""
    app_logger.info(f"Loading session: {session_id}")
    try:
        summarizer = get_summarizer()
        if summarizer.load_session_by_id(session_id):
            # Try to get session statistics from database, but handle if method doesn't exist
            session_stats = None
            try:
                if hasattr(summarizer.vector_db, 'get_session_stats'):
                    session_stats = summarizer.vector_db.get_session_stats(session_id)
                    app_logger.info(f"Retrieved session stats: {session_stats}")
                else:
                    app_logger.warning("VectorDatabase doesn't have get_session_stats method")
            except Exception as stats_error:
                app_logger.error(f"Error getting session stats: {str(stats_error)}")
                session_stats = None
            
            # Get current session data
            session_data = {
                'session_id': summarizer.current_session.session_id,
                'pdf_path': summarizer.current_session.pdf_path,
                'created_at': summarizer.current_session.created_at,
                'total_chunks': len(summarizer.document_chunk_objects),
                'summary': summarizer.current_session.summary,
                'document_path': getattr(summarizer, 'document_path', summarizer.current_session.pdf_path),
                'stats': None  # Initialize stats as None
            }
            
            # Try to format stats if available
            if session_stats:
                formatted_stats = format_stats_for_frontend(session_stats)
                session_data['stats'] = formatted_stats
            else:
                # Create basic stats from available data if no database stats
                app_logger.info("Creating basic stats from session data")
                try:
                    if summarizer.current_session.summary:
                        summary_word_count = len(summarizer.current_session.summary.split())
                        session_data['stats'] = {
                            'original': {
                                'page_count': 0,
                                'word_count': 0,
                                'char_count': 0,
                                'reading_time': 0
                            },
                            'summary': {
                                'page_count': 0,
                                'word_count': summary_word_count,
                                'char_count': len(summarizer.current_session.summary),
                                'reading_time': calculate_reading_time(summary_word_count)
                            },
                            'processing_time': 0
                        }
                except Exception as basic_stats_error:
                    app_logger.error(f"Error creating basic stats: {str(basic_stats_error)}")
                    session_data['stats'] = None
            
            app_logger.info(f"Loaded session successfully: {session_id}")
            return jsonify({
                'success': True,
                'session': session_data
            })
        else:
            app_logger.error(f"Session not found: {session_id}")
            return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        app_logger.error(f"Failed to load session {session_id}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to load session: {str(e)}'}), 500

@app.route('/session/<session_id>/history')
def get_qa_history(session_id):
    """Get Q&A history for session"""
    try:
        summarizer = get_summarizer()
        
        # Check if the get_qa_history method exists and handle gracefully
        history = []
        try:
            if hasattr(summarizer.vector_db, 'get_qa_history'):
                history = summarizer.vector_db.get_qa_history(session_id, limit=50)
                app_logger.info(f"Retrieved {len(history)} Q&A records for session {session_id}")
            else:
                app_logger.warning("VectorDatabase doesn't have get_qa_history method")
                history = []
        except Exception as history_error:
            app_logger.error(f"Error getting Q&A history: {str(history_error)}")
            history = []
        
        return jsonify([{
            'question': qa.question,
            'answer': qa.answer,
            'confidence': qa.confidence,
            'source_chunks': qa.source_chunks,
            'processing_time': qa.processing_time
        } for qa in history])
    except Exception as e:
        app_logger.error(f"Failed to get Q&A history: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get history'}), 500

@app.route('/session/<session_id>/delete', methods=['DELETE'])
def delete_session(session_id):
    """Delete session"""
    app_logger.info(f"Deleting session: {session_id}")
    try:
        summarizer = get_summarizer()
        summarizer.delete_session(session_id)
        app_logger.info(f"Session deleted successfully: {session_id}")
        return jsonify({'success': True})
    except Exception as e:
        app_logger.error(f"Failed to delete session {session_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated summary files"""
    app_logger.info(f"Download request for: {filename}")
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        app_logger.error(f"Download failed for {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/debug/stats/<task_id>')
def debug_stats(task_id):
    """Debug endpoint to check statistics format"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    task_status = processing_status[task_id]
    stats = task_status.get('result', {}).get('stats') if task_status.get('result') else None
    
    return jsonify({
        'task_id': task_id,
        'task_status': task_status.get('status'),
        'raw_stats': stats,
        'stats_type': type(stats).__name__,
        'stats_keys': list(stats.keys()) if isinstance(stats, dict) else None
    })
    
@app.route('/debug/status')
def debug_status():
    """Debug endpoint to check application status"""
    return jsonify({
        'active_sessions': list(active_sessions.keys()),
        'processing_status': {k: v for k, v in processing_status.items()},
        'active_tasks': list(active_tasks.keys()),
        'timestamp': datetime.now().isoformat()
    })

# SocketIO events
@socketio.on('join_task')
def join_task(data):
    """Join task room for progress updates"""
    task_id = data.get('task_id')
    if task_id:
        room_name = f"task_{task_id}"
        join_room(room_name)
        socketio_logger.info(f"Client joined room: {room_name}")
        emit('joined_task', {'task_id': task_id, 'room': room_name})

@socketio.on('leave_task')
def leave_task(data):
    """Leave task room"""
    task_id = data.get('task_id')
    if task_id:
        room_name = f"task_{task_id}"
        leave_room(room_name)
        socketio_logger.info(f"Client left room: {room_name}")

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    socketio_logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'message': 'Connected to PDF Analyzer',
        'client_id': request.sid
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    socketio_logger.info(f"Client disconnected: {request.sid}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    app_logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app_logger.error(f"500 error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    app_logger.info("Starting PDF Analyzer Web Application...")
    app_logger.info(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    app_logger.info(f"Access at: http://localhost:5000")
    app_logger.info(f"Python 3.12 compatible mode (threading)")
    
    # Check if template exists
    template_path = templates_dir / 'index.html'
    if not template_path.exists():
        app_logger.warning("index.html template not found - you'll need to create it!")
    
    try:
        # Use threading mode which is compatible with Python 3.12
        socketio.run(
            app, 
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            use_reloader=False,  # Disable reloader in threading mode
            allow_unsafe_werkzeug=True  # Allow Werkzeug in production-like settings
        )
    except Exception as e:
        app_logger.error(f"Failed to start server: {str(e)}", exc_info=True)