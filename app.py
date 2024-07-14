import streamlit as st
import PyPDF2
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a more suitable model for code generation
MODEL_NAME = "microsoft/CodeGPT-small-py"

@st.cache_resource()
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

def generate_code(text, language='python'):
    try:
        prompt = f"Convert this scientific methodology to {language} code:\n\n{text}\n\n{language} code:"
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=1000, 
                num_return_sequences=1, 
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated code part
        code_start = generated_code.find(f"{language} code:")
        if code_start != -1:
            generated_code = generated_code[code_start + len(f"{language} code:"):]
        
        return generated_code.strip()
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        return "Error: Failed to generate code. Please try again with a different input."

def main():
    st.title("Paper to Code Converter")

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
            with st.spinner("Converting to code... This may take a moment."):
                try:
                    output_code = generate_code(input_text, language.lower())
                    st.header("Output: Code Implementation")
                    st.code(output_code, language=language.lower())
                except Exception as e:
                    logger.error(f"Error processing input: {str(e)}")
                    st.error("An error occurred while processing your input. Please try again.")
        else:
            st.warning("Please enter some text or upload a PDF file.")

if __name__ == "__main__":
    main()
