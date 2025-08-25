#!/usr/bin/env python3
"""
Test script to verify AI Chat API with different questions
"""

import requests
import json
import io
from PyPDF2 import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf():
    """Create a simple test PDF with some content"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Test Document")
    c.drawString(100, 720, "This is a sample PDF for testing AI chat functionality.")
    c.drawString(100, 690, "It contains information about cats, dogs, and programming.")
    c.drawString(100, 660, "Cats are independent animals that love to sleep.")
    c.drawString(100, 630, "Dogs are loyal companions that enjoy playing fetch.")
    c.drawString(100, 600, "Programming is the art of solving problems with code.")
    c.save()
    buffer.seek(0)
    return buffer

def test_chat_api():
    """Test the AI chat API with different questions"""
    
    # Create test PDF
    pdf_buffer = create_test_pdf()
    
    questions = [
        "What is this document about?",
        "Tell me about cats",
        "What does it say about dogs?",
        "Does it mention programming?",
        "How many animals are discussed?"
    ]
    
    print("Testing AI Chat API with different questions...")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        
        try:
            # Reset buffer position
            pdf_buffer.seek(0)
            
            # Prepare request
            files = {'file': ('test.pdf', pdf_buffer, 'application/pdf')}
            data = {'question': question}
            
            # Make request
            response = requests.post(
                'http://localhost:8000/api/ai-chat-pdf',
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('message', 'No message')
                print(f"Answer: {answer[:200]}...")
                print(f"Pages analyzed: {result.get('pages_analyzed', 'N/A')}")
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Exception: {e}")
    
    print("\n" + "=" * 60)
    print("If answers vary by question, the AI chat is working correctly!")

if __name__ == "__main__":
    test_chat_api() 