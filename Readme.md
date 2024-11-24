# PDF Query Assistant

An intelligent PDF document analysis tool built with Streamlit and Google's Generative AI.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/sayan-bhattacharya/RAG_soup.git
cd RAG_soup
```

2. Create and activate virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scriptsctivate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

5. Update the configuration files with your API keys

6. Run the application:
```bash
streamlit run app.py
```

## Features

- PDF text extraction and processing
- Intelligent question answering
- Dark mode support
- Source document references

