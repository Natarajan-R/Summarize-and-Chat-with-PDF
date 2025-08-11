# Summarize-and-Chat-with-PDF   

# AI-Powered Document Processing

<img width="1849" height="1007" alt="Screenshot from 2025-08-11 13-06-44" src="https://github.com/user-attachments/assets/e9adc757-b2f0-4e9c-b2c2-9cb2aa3861a4" />




A web application that uses locally hosted LLMs to summarize PDF documents and enable interactive Q&A about their content. Built with Python, Flask, and Socket.IO for real-time updates.

## Features

- **PDF Summarization**: Generate comprehensive, executive, technical, or bullet-point summaries
- **Document Q&A**: Ask questions about PDF content and get AI-powered answers
- **Local Processing**: Works with locally hosted LLMs (like Mistral) for privacy
- **Session Management**: Save and revisit document processing sessions
- **Real-time Progress**: Track processing with live updates
- **Statistics Dashboard**: View document metrics and compression ratios

## Technology Stack

- **Backend**: Python, Flask, Socket.IO
- **Frontend**: HTML5, CSS3, JavaScript
- **AI Processing**: Local LLM integration (Ollama compatible)
- **Database**: SQLite for session storage
- **Text Processing**: pdfplumber, FAISS for vector search

## Getting Started

### Prerequisites
- Python 3.12+
- Ollama with Mistral (or other local LLM)
- Node.js (for Socket.IO client)

### Installation
```bash
git clone https://github.com/Natarajan-R/Summarize-and-Chat-with-PDF.git
cd Summarize-and-Chat-with-PDF
pip install -r requirements.txt
