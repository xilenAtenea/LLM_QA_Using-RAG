# LLM for Question Answering Using RAG

## Introduction

This repository contains a language model (LLM) for question answering, designed to assist students, teachers, and collaborators at the Universidad Autónoma de Occidente. This model is trained using public data from the university, such as PDFs and official web pages.

Additionally, this repository is based on another project that explains the implemented LLM in detail. You can find the complete explanation of the model at the following link: [Model Explanation](https://github.com/xilenAtenea/LLM_explanation).

Some minimal changes have been made compared to the notebook available in the explanation repository. Specifically, a `chat_loop` function has been added to allow continuous chat with the model without the need to run the entire script each time a question is asked.

## Data Used

The data used in this project is public and available from the Universidad Autónoma de Occidente. It includes:

- Official university PDFs
- University web pages

## Prerequisites

To run this project, you need to have the following programs installed:

- **Python**: I recommend version 3.12.3
- **Ollama**: You need to download it from the official [Ollama repository](https://github.com/ollama/ollama) and install it locally on your computer

## Getting Started

Follow these steps to run the code locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xilenAtenea/LLM-for-Question-answering.git
   cd LLM-for-Question-answering

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`

3. **Install the requirements:**
    ```bash
    pip install -r requirements.txt

4. **Download and install Ollama:**
Visit the official [Ollama repository](https://github.com/ollama/ollama) and follow the instructions for local installation.

5. **Adjust the information according to your objectives:**
Place your own PDFs inside a folder named "info" or use only web pages.
Ensure that the loaders used in the code fit your data sources.

6. **Pull used models**
    ```bash
    ollama pull nomic-embed-text
    ollama pull llama3

7. **Run the code:**
    ```bash
    python qa_llm.py

## Repository Structure

- `main.py`: Main script to run the language model.
- `requirements.txt`: List of dependencies needed to run the project.
- `info/`: Folder where you should place your PDF files (if you choose to use PDFs).

## References
The initial code was based on this [link](https://medium.com/@Sanjjushri/rag-pdf-q-a-using-llama-2-in-8-steps-021a7dbe26e1). It is important to note that one of the key differences in this implementation is the incorporation of Llama3. A way was found to integrate it with the existing code, which improved the performance and functionality of the model. 

For more details on how the model works, or if you are just curious, I invite you to visit my [explanation repository](https://github.com/xilenAtenea/LLM_explanation) about this LLM implementation with Llama 3.

## <b> Let's Connect!!</b>

<br>
<div align='left'>

<ul>

<!--icons and links-->
<p align="center">
<a href="https://www.linkedin.com/in/atenea-rojas" target="blank"><img align="center" src="https://user-images.githubusercontent.com/88904952/234979284-68c11d7f-1acc-4f0c-ac78-044e1037d7b0.png" alt="linkedin" height="50" width="50" /></a>
<!--<a href="" target="blank"><img align="center" src="https://user-images.githubusercontent.com/88904952/234981169-2dd1e58f-4b7e-468c-8213-034ba62156c3.png" alt="instagram" height="50" width="50" /></a>-->
<a href="https://discordapp.com/users/558813893422612541" target="blank"><img align="center" src="https://user-images.githubusercontent.com/88904952/234982627-019fd336-6248-453c-9b05-97c13fd1d207.png" alt="discord" height="50" width="50" /></a>
  
</p>

