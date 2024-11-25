import time
import os
import psutil
import shutil
import requests.exceptions
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Union
from pathlib import Path
import logging
import validators
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import RAG_app

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_remove_directory(path):
  """Safely remove a directory in Windows."""
  max_attempts = 3
  attempt = 0

  while attempt < max_attempts:
      try:
          if os.path.exists(path):
              # Force close any open handles
              for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                  try:
                      for file in proc.open_files():
                          if str(path) in file.path:
                              proc.kill()
                  except (psutil.NoSuchProcess, psutil.AccessDenied):
                      pass

              time.sleep(1)
              shutil.rmtree(path, ignore_errors=True)
          return True
      except Exception as e:
          logger.warning(f"Attempt {attempt + 1} failed to remove directory: {str(e)}")
          attempt += 1
          time.sleep(1)

  return False

def get_vector_store(text_chunks):
  """Create and save vector store from text chunks."""
  try:
      if not text_chunks:
          raise ValueError("No text chunks provided")

      # Use temp directory as backup
      temp_dir = os.path.join(os.environ.get('TEMP', './'), 'faiss_index')
      index_path = Path("./faiss_index")

      # Try primary location first, then fallback to temp
      locations = [index_path, Path(temp_dir)]

      for loc in locations:
          try:
              if not safe_remove_directory(str(loc)):
                  continue

              os.makedirs(str(loc), exist_ok=True)

              embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
              logger.info(f"Creating vector store with {len(text_chunks)} chunks")

              vector_store = FAISS.from_texts(
                  texts=text_chunks,
                  embedding=embeddings
              )

              vector_store.save_local(str(loc))
              st.session_state['vector_store'] = vector_store
              logger.info(f"Vector store saved successfully to {loc}")

              # Verify store
              test_store = FAISS.load_local(
                  str(loc),
                  embeddings,
                  allow_dangerous_deserialization=True
              )
              test_results = test_store.similarity_search("test", k=1)
              logger.info(f"Vector store verified with {len(test_results)} results")

              return True

          except Exception as e:
              logger.warning(f"Failed to use location {loc}: {str(e)}")
              continue

      raise Exception("Failed to create vector store in any location")

  except Exception as e:
      logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
      st.error(f"Error creating vector store: {str(e)}")
      return False

def process_documents(pdf_docs):
  """Process uploaded PDF documents."""
  try:
      all_chunks = []
      for pdf in pdf_docs:
          doc_name = pdf.name
          try:
              raw_text = RAG_app.get_pdf_text([pdf])
              if not raw_text:
                  st.warning(f"No text could be extracted from {doc_name}")
                  continue

              processed_text = RAG_app.preprocess_text(raw_text)
              chunks = RAG_app.get_text_chunks(processed_text, f"PDF: {doc_name}")

              if chunks:
                  all_chunks.extend(chunks)
                  logger.info(f"Successfully processed {doc_name}")
              else:
                  st.warning(f"No valid chunks created from {doc_name}")

          except Exception as e:
              logger.error(f"Error processing {doc_name}: {str(e)}")
              st.warning(f"Error processing {doc_name}: {str(e)}")
              continue

      if not all_chunks:
          st.error("No valid content could be extracted from any documents")
          return False

      success = get_vector_store(all_chunks)
      if success:
          st.session_state['documents_processed'] = True
          return True

      return False

  except Exception as e:
      logger.error(f"Error during document processing: {str(e)}", exc_info=True)
      st.error(f"Error processing documents: {str(e)}")
      return False

def scrape_website(url: str) -> str:
  """Scrape content from a website with improved handling of protected sites."""
  try:
      headers = {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.5',
          'Accept-Encoding': 'gzip, deflate, br',
          'Connection': 'keep-alive',
      }

      time.sleep(2)
      session = requests.Session()
      response = session.get(url, headers=headers, timeout=10)
      response.raise_for_status()

      soup = BeautifulSoup(response.text, 'html.parser')

      # Remove unwanted elements
      for element in soup(['script', 'style', 'header', 'footer', 'nav']):
          element.decompose()

      main_content = soup.find(['main', 'article', 'div[role="main"]']) or soup

      paragraphs = []
      for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
          text = elem.get_text().strip()
          if text:
              paragraphs.append(text)

      if not paragraphs:
          raise ValueError("No content could be extracted from the webpage")

      return f"\n\n--- Website: {url} ---\n" + "\n".join(paragraphs)

  except requests.exceptions.HTTPError as e:
      if e.response.status_code == 403:
          logger.warning(f"Access denied to {url}")
          st.warning(f"Unable to access {url}. Try downloading as PDF instead.")
      return ""
  except Exception as e:
      logger.error(f"Error scraping website {url}: {str(e)}")
      st.error(f"Error accessing {url}: {str(e)}")
      return ""

def process_mixed_sources(pdfs: List[str] = None, urls: List[str] = None) -> bool:
  """Process both PDFs and website content together."""
  try:
      all_chunks = []

      # Process PDFs
      if pdfs:
          for pdf in pdfs:
              try:
                  raw_text = RAG_app.get_pdf_text([pdf])
                  if raw_text:
                      processed_text = RAG_app.preprocess_text(raw_text)
                      chunks = RAG_app.get_text_chunks(processed_text, f"PDF: {pdf.name}")
                      if chunks:
                          all_chunks.extend(chunks)
                          logger.info(f"Successfully processed PDF: {pdf.name}")
              except Exception as e:
                  logger.error(f"Error processing PDF {pdf.name}: {str(e)}")
                  st.warning(f"Error processing PDF {pdf.name}: {str(e)}")

      # Process URLs
      if urls:
          for url in urls:
              try:
                  web_text = scrape_website(url)
                  if web_text:
                      processed_text = RAG_app.preprocess_text(web_text)
                      chunks = RAG_app.get_text_chunks(processed_text, f"URL: {url}")
                      if chunks:
                          all_chunks.extend(chunks)
                          logger.info(f"Successfully processed URL: {url}")
              except Exception as e:
                  logger.error(f"Error processing URL {url}: {str(e)}")
                  st.warning(f"Error processing URL {url}: {str(e)}")

      if not all_chunks:
          st.error("No valid content could be extracted from any source")
          return False

      success = get_vector_store(all_chunks)
      return success

  except Exception as e:
      logger.error(f"Error in process_mixed_sources: {str(e)}")
      st.error(f"Error processing sources: {str(e)}")
      return False

def enhanced_prompt_template() -> PromptTemplate:
  template = """
  You are an intelligent assistant analyzing multiple documents.

  Context: {context}
  Question: {question}

  Guidelines:
  1. Synthesize information across documents
  2. Identify patterns and relationships
  3. Provide specific examples and quotes
  4. Acknowledge conflicts between sources
  5. Structure complex answers clearly

  Detailed Answer:
  """

  return PromptTemplate(
      template=template,
      input_variables=["context", "question"]
  )

def enhanced_main():
  st.set_page_config(page_title="Enhanced Document Analysis Assistant", layout="wide")
  st.title("🔍 Document Analysis Assistant")

  with st.sidebar:
      st.title("📚 Upload Sources")

      pdf_docs = st.file_uploader(
          "Upload PDFs",
          type=['pdf'],
          accept_multiple_files=True,
          help="Upload one or more PDF documents"
      )

      urls = st.text_area(
          "Enter URLs (one per line)",
          help="Enter website URLs to include in the analysis"
      )
      url_list = [url.strip() for url in urls.split('\n') if url.strip()] if urls else []

      if st.button("🔄 Process Sources", disabled=not (pdf_docs or url_list)):
          with st.spinner("Processing sources..."):
              if process_mixed_sources(pdf_docs, url_list):
                  st.success("Sources processed successfully!")

  if st.session_state.get('documents_processed', False):
      user_question = st.text_input("Ask a question about your documents:")

      if user_question:
          try:
              chain = RetrievalQA.from_chain_type(
                  llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7),
                  chain_type="stuff",
                  retriever=st.session_state['vector_store'].as_retriever(
                      search_type="similarity",
                      search_kwargs={"k": 6}
                  ),
                  return_source_documents=True,
                  chain_type_kwargs={"prompt": enhanced_prompt_template()}
              )

              response = chain(user_question)
              st.write(response['result'])

          except Exception as e:
              logger.error(f"Error processing question: {str(e)}")
              st.error("Error processing your question. Please try again.")

  with st.expander("ℹ️ Tips"):
      st.markdown("""
      ### Tips for Best Results
      - Upload multiple related documents
      - Use specific, focused questions
      - Try comparing information across sources
      - Look for patterns in the responses
      """)

if __name__ == "__main__":
  enhanced_main()