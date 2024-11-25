# enhanced_pdf_assistant.py
import time
import requests.exceptions
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Union
import RAG_app # your original file
from langchain.prompts import PromptTemplate
import logging
import validators

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_website(url: str) -> str:
  """Scrape content from a website with improved handling of protected sites."""
  try:
      # Enhanced headers to appear more like a real browser
      headers = {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.5',
          'Accept-Encoding': 'gzip, deflate, br',
          'Connection': 'keep-alive',
          'Upgrade-Insecure-Requests': '1',
          'Sec-Fetch-Dest': 'document',
          'Sec-Fetch-Mode': 'navigate',
          'Sec-Fetch-Site': 'none',
          'Sec-Fetch-User': '?1',
          'Cache-Control': 'max-age=0'
      }

      # Add delay to respect rate limits
      time.sleep(2)

      session = requests.Session()
      response = session.get(url, headers=headers, timeout=10)
      response.raise_for_status()

      soup = BeautifulSoup(response.text, 'html.parser')

      # Remove unwanted elements
      for element in soup(['script', 'style', 'header', 'footer', 'nav']):
          element.decompose()

      # More targeted content extraction based on common content containers
      main_content = soup.find(['main', 'article', 'div[role="main"]']) or soup

      # Extract text with better formatting
      paragraphs = []
      for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
          text = elem.get_text().strip()
          if text:
              paragraphs.append(text)

      formatted_text = f"\n\n--- Website: {url} ---\n" + "\n".join(paragraphs)

      if not paragraphs:
          raise ValueError("No content could be extracted from the webpage")

      return formatted_text

  except requests.exceptions.HTTPError as e:
      if e.response.status_code == 403:
          alternative_message = f"""
          Unable to access {url} directly due to website restrictions. 
          For Indeed.com and similar protected websites, try these alternatives:
          1. Use the PDF upload feature instead
          2. Copy-paste the content manually into a text file
          3. Use publicly available API if available
          """
          logger.warning(alternative_message)
          st.warning(alternative_message)
          return ""
  except Exception as e:
      logger.error(f"Error scraping website {url}: {str(e)}")
      st.error(f"Error accessing {url}: {str(e)}")
      return ""
def enhanced_prompt_template() -> PromptTemplate:
  """Create an enhanced prompt template for better context understanding."""
  template = """
  You are an intelligent assistant with comprehensive knowledge of the provided documents.
  Analyze the following context thoroughly and provide a detailed, creative response.

  Guidelines:
  1. Synthesize information across multiple documents when relevant
  2. Identify patterns and relationships between different sources
  3. Provide specific examples and quotes to support your answers
  4. When appropriate, offer insights that combine information from different sources
  5. If information conflicts between sources, acknowledge and explain the differences

  Context: {context}

  Question: {question}

  Additional Instructions:
  - If the question requires comparing information across documents, explicitly mention the connections
  - If relevant information seems missing, acknowledge what additional information would be helpful
  - Provide confidence levels for your answers when appropriate
  - Structure complex answers with clear sections and bullet points

  Detailed Answer:
  """

  return PromptTemplate(
      template=template,
      input_variables=["context", "question"]
  )

def process_mixed_sources(pdfs: List[str] = None, urls: List[str] = None) -> str:
  """Process both PDFs and website content together."""
  combined_text = ""

  # Process PDFs
  if pdfs:
      pdf_text = RAG_app.get_pdf_text(pdfs)
      combined_text += pdf_text

  # Process URLs
  if urls:
      for url in urls:
          try:
              web_text = scrape_website(url)
              combined_text += web_text
          except Exception as e:
              st.error(f"Error processing URL {url}: {str(e)}")
              continue

  return combined_text

def enhanced_main():
  """Enhanced main function with additional features."""
  st.set_page_config(page_title="Enhanced PDF & Web Query Assistant", layout="wide")

  st.title("🔍 Search your query below")
  # Sidebar for document inputs
  with st.sidebar:
      st.title("📚 Source Inputs")

      # PDF Upload
      pdf_docs = st.file_uploader(
          "Upload PDFs",
          type=['pdf'],
          accept_multiple_files=True,
          help="Upload one or more PDF documents"
      )

      # URL Input
      urls = st.text_area(
          "Enter URLs (one per line)",
          help="Enter website URLs to include in the analysis"
      )
      url_list = [url.strip() for url in urls.split('\n') if url.strip()] if urls else []

      # Process Button
      if st.button("🔄 Process All Sources", disabled=not (pdf_docs or url_list)):
          try:
              with st.spinner("Processing all sources..."):
                  combined_text = process_mixed_sources(pdf_docs, url_list)
                  chunks = RAG_app.get_text_chunks(combined_text)
                  if RAG_app.get_vector_store(chunks):
                      st.session_state['documents_processed'] = True
                      st.success("All sources processed successfully!")
          except Exception as e:
              st.error(f"Error during processing: {str(e)}")

  # Main area for queries
  if st.session_state.get('documents_processed', False):
      
      user_question = st.text_input(
          "🤔 What would you like to know?",
          placeholder="Ask anything about your documents..."
      )

      if user_question:
          # Override the original prompt template with enhanced version
          RAG_app.get_conversational_chain = lambda: RAG_app.RetrievalQA.from_chain_type(
              llm=RAG_app.ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7),
              chain_type="stuff",
              retriever=st.session_state['vector_store'].as_retriever(
                  search_type="similarity",
                  search_kwargs={"k": 6}  # Increased context window
              ),
              return_source_documents=True,
              chain_type_kwargs={"prompt": enhanced_prompt_template()}
          )

          RAG_app.user_input(user_question, st.session_state.get('dark_mode', False))

  # Additional features and tips
  with st.expander("🌟 Advanced Features"):
      st.markdown("""
      ### Enhanced Capabilities
      1. **Multi-Source Analysis**: Combines information from PDFs and websites
      2. **Creative Synthesis**: Generates insights across multiple documents
      3. **Source Tracking**: Maintains clear references to source materials
      4. **Confidence Levels**: Indicates reliability of information
      5. **Pattern Recognition**: Identifies relationships between different sources

      ### Tips for Best Results
      - Combine different types of sources for comprehensive analysis
      - Use specific questions to get detailed answers
      - Try comparing information across different sources
      - Look for patterns and connections in the responses
      """)

if __name__ == "__main__":
  enhanced_main()