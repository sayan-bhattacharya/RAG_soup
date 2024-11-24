Chat with PDF using Gemini 💬

This application allows users to interact with PDF documents using natural language queries. Powered by Google Generative AI and LangChain, it extracts relevant information from uploaded PDFs and provides concise answers in a conversational format. The app also provides references from the PDFs to support its answers.

Features

	•	📄 PDF Upload: Upload one or multiple PDFs for processing.
	•	🧠 AI-Powered Q&A: Ask questions about the uploaded PDFs and receive detailed answers.
	•	📑 Contextual References: View the specific text chunks used to generate the answers.
	•	🔗 Bullet-Point Outputs: Answers are structured in a clean, bullet-point format for better readability.
	•	🛠 Powered By:
	•	Streamlit for the user interface.
	•	Google Generative AI for embeddings and conversational chains.
	•	FAISS for vector-based similarity search.
	•	LangChain for efficient query handling.

How It Works

	1.	Upload PDFs: Use the file uploader to select the PDFs you want to analyze.
	2.	Process PDFs: The app processes the PDFs, splits them into manageable chunks, and stores them in a vector database.
	3.	Ask Questions: Enter your question in the input box, and the app will fetch the most relevant text from the PDFs and generate an answer.
	4.	View References: Click the “Show References” button to see the specific text chunks from which the answer was derived.

Installation

Follow these steps to set up and run the application locally:

1. Clone the Repository

git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf

2. Create a Virtual Environment

python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up API Keys

Create a .env file in the root directory and add your Google Generative AI API key:

GOOGLE_API_KEY=your_google_api_key

5. Run the Application

streamlit run RAG_app.py

The app will launch in your default browser. If it doesn’t, visit the URL provided in the terminal, typically http://localhost:8501.

Deployment

Deploy on Streamlit Cloud

	1.	Push the code to a GitHub repository.
	2.	Visit Streamlit Cloud and connect your GitHub account.
	3.	Deploy the app by selecting the repository and specifying the file path (RAG_app.py).

For detailed steps, refer to the Streamlit Cloud Documentation.

Dependencies

The app requires the following Python libraries:
	•	streamlit
	•	google-generativeai
	•	langchain
	•	langchain_community
	•	faiss-cpu
	•	PyPDF2
	•	python-dotenv

All dependencies are listed in the requirements.txt file.

Usage

	1.	PDF Processing:
	•	Upload PDFs via the sidebar.
	•	Click “Submit & Process” to split and store the content in a vector database.
	2.	Ask Questions:
	•	Enter your question in the input field.
	•	Click “Show References” to view the text used for generating the answer.
	3.	Example Questions:
	•	“What is the summary of the first document?”
	•	“What are the main points discussed about AI?”
	•	“Who are the authors mentioned in the document?”

Screenshots

Contributing

Contributions are welcome! To contribute:
	1.	Fork the repository.
	2.	Create a new branch:

git checkout -b feature-name


	3.	Make your changes and commit:

git commit -m "Added new feature"


	4.	Push to your fork:

git push origin feature-name


	5.	Open a Pull Request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For questions or feedback, feel free to reach out:
	•	Author: Sayan Bhattacharya
	•	GitHub: sayan-bhattacharya