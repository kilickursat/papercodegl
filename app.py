import streamlit as st
import PyPDF2
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a smaller model
MODEL_NAME = "distilgpt2"

@st.cache_data(allow_output_mutation=True)
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

tokenizer, model = load_model()

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def generate_text(prompt, max_length=500):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, temperature=0.7, top_p=0.95, do_sample=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return ""

def generate_code(text, language='python'):
    prompt = f"Convert the following scientific methodology to {language} code:\n\n{text}\n\nGenerated {language} code:"
    generated_text = generate_text(prompt)
    
    # Extract only the generated code part
    code_start = generated_text.find(f"Generated {language} code:")
    if code_start != -1:
        generated_text = generated_text[code_start + len(f"Generated {language} code:"):]
    
    return generated_text.strip()

def summarize_text(text):
    prompt = f"Summarize the following scientific methodology:\n\n{text}\n\nSummary:"
    summary = generate_text(prompt, max_length=200)
    
    # Extract only the summary part
    summary_start = summary.find("Summary:")
    if summary_start != -1:
        summary = summary[summary_start + len("Summary:"):]
    
    return summary.strip()

def main():
    st.title("Optimized Paper to Code Converter")

    if tokenizer is None or model is None:
        st.error("Failed to load the model. Please try again later.")
        return

    st.header("Input: Scientific Paper Methodology")
    input_method = st.radio("Choose input method:", ("Text", "PDF Upload"))

    if input_method == "Text":
        input_text = st.text_area("Paste your paper's methodology here:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            input_text = extract_text_from_pdf(uploaded_file)
            st.text_area("Extracted text:", value=input_text, height=200)
        else:
            input_text = ""

    language = st.selectbox("Select output programming language:", ("Python", "Java", "C++"))

    if st.button("Convert to Code"):
        if input_text:
            with st.spinner("Processing... This may take a moment."):
                try:
                    output_code = generate_code(input_text, language.lower())
                    summary = summarize_text(input_text)

                    st.header("Output: Code Implementation")
                    st.code(output_code, language=language.lower())
                    
                    st.header("Summary of Methodology")
                    st.write(summary)
                except Exception as e:
                    logger.error(f"Error processing input: {str(e)}")
                    st.error("An error occurred while processing your input. Please try again.")
        else:
            st.warning("Please enter some text or upload a PDF file.")

if __name__ == "__main__":
    main()
