import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import google.generativeai as genai
import re
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'documents_processed' not in st.session_state:
  st.session_state['documents_processed'] = False
if 'processed_files' not in st.session_state:
  st.session_state['processed_files'] = []
if 'vector_store' not in st.session_state:
  st.session_state['vector_store'] = None

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
  raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
genai.configure(api_key=api_key)

def verify_pdf(pdf_file):
  """Verify if the PDF file is valid and readable."""
  try:
      reader = PdfReader(pdf_file)
      if len(reader.pages) == 0:
          return False, "PDF file contains no pages."
      pdf_file.seek(0)  # Reset file pointer
      return True, f"Valid PDF with {len(reader.pages)} pages."
  except Exception as e:
      logger.error(f"Error verifying PDF {pdf_file.name}: {str(e)}")
      return False, f"Invalid PDF file: {str(e)}"

def preprocess_text(text):
  """Clean and prepare text for better processing."""
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'[^a-zA-Z0-9\s.,!?()-]', '', text)
  text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
  return text.strip()

def get_pdf_text(pdf_docs):
  """Extract and preprocess text from uploaded PDFs."""
  if not pdf_docs:
      st.error("Please upload at least one PDF document.")
      return ""

  combined_text = ""
  st.session_state['processed_files'] = []

  for pdf in pdf_docs:
      try:
          with st.spinner(f"Processing {pdf.name}..."):
              is_valid, message = verify_pdf(pdf)
              if not is_valid:
                  st.error(f"Error with {pdf.name}: {message}")
                  continue

              pdf_reader = PdfReader(pdf)
              total_pages = len(pdf_reader.pages)
              progress_text = f"Processing {pdf.name}"
              progress_bar = st.progress(0, text=progress_text)

              for page_num, page in enumerate(pdf_reader.pages, 1):
                  progress = page_num / total_pages
                  progress_bar.progress(progress, text=f"{progress_text} - Page {page_num}/{total_pages}")

                  page_text = page.extract_text()
                  if page_text:
                      combined_text += f"\n\n--- Document: {pdf.name} - Page {page_num} ---\n"
                      combined_text += preprocess_text(page_text)
                  else:
                      st.warning(f"No text found on page {page_num} of {pdf.name}")

              progress_bar.empty()
              st.session_state['processed_files'].append(pdf.name)
              st.success(f"Successfully processed {pdf.name} ({total_pages} pages)")
              logger.info(f"Successfully processed {pdf.name}")

      except Exception as e:
          st.error(f"Error processing {pdf.name}: {str(e)}")
          logger.error(f"Error processing {pdf.name}: {str(e)}")
          continue

  if not combined_text.strip():
      st.error("No text could be extracted from any of the uploaded PDFs.")
      return ""

  return combined_text

def get_text_chunks(text):
  """Split the extracted text into smaller, more meaningful chunks."""
  try:
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,
          chunk_overlap=200,
          length_function=len,
          separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
      )
      chunks = text_splitter.split_text(text)

      # Debug logging
      logger.info(f"Created {len(chunks)} text chunks")
      logger.debug(f"First chunk sample: {chunks[0][:200] if chunks else 'No chunks created'}")

      return chunks

  except Exception as e:
      logger.error(f"Error creating text chunks: {str(e)}", exc_info=True)
      st.error(f"Failed to create text chunks: {str(e)}")
      return []
    
def get_vector_store(text_chunks):
  """Create and save vector store from text chunks."""
  try:
      if not text_chunks:
          raise ValueError("No text chunks provided")

      if os.path.exists("faiss_index"):
          shutil.rmtree("faiss_index")

      embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

      logger.info(f"Creating vector store with {len(text_chunks)} chunks")

      vector_store = FAISS.from_texts(
          texts=text_chunks,
          embedding=embeddings
      )

      # Save vector store
      vector_store.save_local("faiss_index")
      st.session_state['vector_store'] = vector_store

      # Verify vector store with safe deserialization
      test_store = FAISS.load_local(
          "faiss_index", 
          embeddings,
          allow_dangerous_deserialization=True
      )
      test_results = test_store.similarity_search("test", k=1)
      logger.info(f"Vector store creation successful. Test search returned {len(test_results)} results")

      return True

  except Exception as e:
      logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
      st.error(f"Error creating vector store: {str(e)}")
      return False

def get_conversational_chain():
  """Create the conversational chain for question answering."""
  prompt_template = """
  Use the following pieces of context to answer the question at the end. 
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  Context: {context}

  Question: {question}

  Answer the question based on the context provided. Include relevant quotes from the context to support your answer.
  """

  try:
      model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

      prompt = PromptTemplate(
          template=prompt_template,
          input_variables=["context", "question"]
      )

      # Load vector store with allow_dangerous_deserialization=True
      if os.path.exists("faiss_index"):
          embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
          vector_store = FAISS.load_local(
              "faiss_index", 
              embeddings,
              allow_dangerous_deserialization=True  # Added this parameter
          )
          st.session_state['vector_store'] = vector_store
      else:
          raise ValueError("Vector store not found. Please process documents first.")

      retriever = vector_store.as_retriever(
          search_type="similarity",
          search_kwargs={"k": 4}
      )

      chain = RetrievalQA.from_chain_type(
          llm=model,
          chain_type="stuff",
          retriever=retriever,
          return_source_documents=True,
          chain_type_kwargs={"prompt": prompt}
      )

      # Verify chain creation
      logger.info("Conversational chain created successfully")
      return chain

  except Exception as e:
      logger.error(f"Error creating conversational chain: {str(e)}")
      raise
  
def user_input(user_question, dark_mode):
  """Process user input and generate response."""
  try:
      if not st.session_state.get('documents_processed', False):
          st.warning("Please process documents before asking questions.")
          return

      chain = get_conversational_chain()

      with st.spinner("Generating response..."):
          response = chain({"query": user_question})

          logger.info("Response received from chain")

          if not response or 'result' not in response:
              st.error("Failed to get a response from the chain")
              return

          result = response['result']
          source_docs = response.get('source_documents', [])

          # Display the main response
          st.markdown("### Response")
          st.markdown(result)

          # Display source documents
          if source_docs:
              with st.expander("📚 Source Documents"):
                  for i, doc in enumerate(source_docs, 1):
                      st.markdown(f"**Source {i}:**")
                      st.markdown(doc.page_content)
                      st.markdown("---")

  except Exception as e:
      st.error(f"❌ Error generating response: {str(e)}")
      logger.error(f"Error processing question: {str(e)}", exc_info=True)


def process_documents(pdf_docs):
  """Process uploaded documents with progress tracking."""
  try:
      # Reset state
      st.session_state['documents_processed'] = False
      st.session_state['processed_files'] = []
      text_chunks = []  # Initialize text_chunks

      # Extract text from PDFs
      with st.spinner("Extracting text from PDFs..."):
          raw_text = get_pdf_text(pdf_docs)
          if not raw_text.strip():
              st.error("No text could be extracted from the PDFs.")
              return False

      # Create text chunks
      with st.spinner("Creating text chunks..."):
          text_chunks = get_text_chunks(raw_text)  # Assign text_chunks here
          if not text_chunks:
              st.error("Failed to create text chunks.")
              return False
          st.success(f"Created {len(text_chunks)} text chunks")
          logger.info(f"Created {len(text_chunks)} text chunks")

      # Create vector store
      with st.spinner("Creating vector store..."):
          success = get_vector_store(text_chunks)
          if success:
              st.session_state['documents_processed'] = True

              # Debug: Test vector store
              test_store = st.session_state.get('vector_store')
              if test_store:
                  try:
                      test_results = test_store.similarity_search("test", k=1)
                      logger.info(f"Vector store test search results: {bool(test_results)}")
                  except Exception as e:
                      logger.error(f"Vector store test failed: {str(e)}")

              return True
          return False

  except Exception as e:
      st.error(f"❌ Error during processing: {str(e)}")
      logger.error(f"Error during document processing: {str(e)}", exc_info=True)
      return False
def main():
  """Main application function."""
  st.set_page_config(page_title="PDF Query Assistant", layout="wide")

  st.title("📄 Intelligent PDF Query Assistant")
  st.subheader("Upload PDFs and ask questions about their content")

  with st.sidebar:
      dark_mode = st.checkbox("🌙 Dark Mode")

      st.title("📂 Document Upload")
      pdf_docs = st.file_uploader(
          "Upload PDFs",
          type=['pdf'],
          accept_multiple_files=True,
          help="Upload one or more PDF documents"
      )

      if pdf_docs:
          st.write("📑 Uploaded files:")
          for doc in pdf_docs:
              st.write(f"- {doc.name} ({doc.size/1024:.1f} KB)")

          col1, col2 = st.columns(2)
          with col1:
              if st.button("📥 Process", disabled=not pdf_docs):
                  process_documents(pdf_docs)

          with col2:
              if st.button("🗑️ Clear"):
                  st.session_state['documents_processed'] = False
                  st.session_state['processed_files'] = []
                  st.session_state['vector_store'] = None
                  if os.path.exists("faiss_index"):
                      shutil.rmtree("faiss_index")
                  st.rerun()

  if st.session_state.get('processed_files'):
      st.sidebar.success(f"Processed {len(st.session_state['processed_files'])} files")
      for file in st.session_state['processed_files']:
          st.sidebar.write(f"✓ {file}")

  if st.session_state.get('documents_processed', False):
      user_question = st.text_input(
          "💭 Ask a question about your documents:", 
          placeholder="Enter your question here..."
      )
      if user_question:
          user_input(user_question, dark_mode)
  else:
      st.info("👆 Please upload and process documents before asking questions.")

  with st.expander("💡 Tips for better results"):
      st.markdown("""
      1. **Document Quality**: Ensure your PDFs contain searchable text
      2. **Question Clarity**: Be specific in your questions
      3. **Multiple Documents**: You can upload multiple PDFs at once
      4. **Processing Time**: Larger documents may take longer to process
      5. **Clear and Retry**: If you encounter issues, try clearing and processing again
      """)

if __name__ == "__main__":
  main()  