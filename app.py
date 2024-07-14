import streamlit as st
import PyPDF2
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Galactica model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "facebook/galactica-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_code(text, language='python'):
    prompt = f"Convert the following scientific methodology to {language} code:\n\n{text}\n\nGenerated {language} code:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated code part
    code_start = generated_code.find(f"Generated {language} code:")
    if code_start != -1:
        generated_code = generated_code[code_start + len(f"Generated {language} code:"):]
    
    return generated_code.strip()

def summarize_text(text):
    prompt = f"Summarize the following scientific methodology:\n\n{text}\n\nSummary:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the summary part
    summary_start = summary.find("Summary:")
    if summary_start != -1:
        summary = summary[summary_start + len("Summary:"):]
    
    return summary.strip()

def main():
    st.title("Galactica-enhanced Paper to Code Converter")

    # Input section
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

    # Language selection
    language = st.selectbox("Select output programming language:", ("Python", "Java", "C++"))

    # Conversion button
    if st.button("Convert to Code"):
        if input_text:
            with st.spinner("Processing... This may take a moment."):
                # Generate code
                output_code = generate_code(input_text, language.lower())
                
                # Summarize the input text
                summary = summarize_text(input_text)

                # Output section
                st.header("Output: Code Implementation")
                st.code(output_code, language=language.lower())
                
                st.header("Summary of Methodology")
                st.write(summary)
        else:
            st.warning("Please enter some text or upload a PDF file.")

if __name__ == "__main__":
    main()
