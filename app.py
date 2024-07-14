import streamlit as st
import PyPDF2
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import re
import ast
import astroid

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

def generate_code(text, language='python', libraries=None):
    try:
        # Include specified libraries in the prompt
        library_prompt = ""
        if libraries:
            library_prompt = f"Use the following libraries: {', '.join(libraries)}.\n"

        prompt = f"""
        Convert this scientific methodology to {language} code:

        Methodology:
        {text}

        {library_prompt}
        Generate a complete and functional {language} implementation of the above methodology.
        Include necessary imports, functions, and a main execution block.
        Add comments to explain key parts of the code.

        {language} code:
        """

        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=2048,
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
        
        # Clean up and post-process the generated code
        generated_code = clean_generated_code(generated_code, language)
        generated_code = post_process_code(generated_code, language)
        
        return generated_code.strip()
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        return "Error: Failed to generate code. Please try again with a different input."

def clean_generated_code(code, language):
    # Remove any text before the first import or def statement
    code_lines = code.split('\n')
    start_index = 0
    for i, line in enumerate(code_lines):
        if line.startswith('import ') or line.startswith('def ') or line.startswith('class '):
            start_index = i
            break
    code = '\n'.join(code_lines[start_index:])
    
    # Remove any text after the last line of code
    code = re.sub(r'\n\s*$', '', code)
    
    return code

def post_process_code(code, language):
    if language.lower() == 'python':
        return post_process_python(code)
    # Add post-processing for other languages here if needed
    return code

def post_process_python(code):
    try:
        # Parse the code to check for syntax errors
        ast.parse(code)
    except SyntaxError as e:
        # If there's a syntax error, try to fix common issues
        code = fix_common_python_errors(code)
    
    try:
        # Use astroid to parse and modify the AST
        module = astroid.parse(code)
        
        # Ensure proper imports
        ensure_imports(module)
        
        # Ensure there's a main block
        ensure_main_block(module)
        
        # Convert the modified AST back to code
        code = module.as_string()
    except Exception as e:
        logger.error(f"Error in post-processing: {str(e)}")
    
    return code

def fix_common_python_errors(code):
    # Fix indentation issues
    lines = code.split('\n')
    fixed_lines = []
    current_indent = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ')):
            fixed_lines.append('    ' * current_indent + stripped)
            current_indent += 1
        elif stripped.startswith(('return', 'break', 'continue')):
            fixed_lines.append('    ' * (current_indent - 1) + stripped)
        else:
            fixed_lines.append('    ' * current_indent + stripped)
    
    # Join the fixed lines
    return '\n'.join(fixed_lines)

def ensure_imports(module):
    # Check if common libraries are used but not imported
    used_modules = set()
    for node in module.body:
        if isinstance(node, astroid.Call):
            if isinstance(node.func, astroid.Name):
                used_modules.add(node.func.name)
    
    # Add missing imports
    common_modules = {'numpy': 'np', 'pandas': 'pd', 'matplotlib.pyplot': 'plt'}
    for module_name, alias in common_modules.items():
        if alias in used_modules and not any(isinstance(n, astroid.Import) and module_name in n.names for n in module.body):
            import_node = astroid.Import(names=[(module_name, alias)])
            module.body.insert(0, import_node)

def ensure_main_block(module):
    # Check if there's a main block, add if missing
    if not any(isinstance(n, astroid.If) and n.test.as_string() == '__name__ == "__main__"' for n in module.body):
        main_block = astroid.If(
            test=astroid.Compare(
                left=astroid.Name(name='__name__'),
                ops=[('==', astroid.Const(value='__main__'))]
            ),
            body=[astroid.Pass()]
        )
        module.body.append(main_block)

def main():
    st.title("Scientific Paper to Code Converter")
    
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
    
    # Allow users to specify libraries
    libraries = st.multiselect(
        "Select libraries to use (optional):",
        ["numpy", "pandas", "matplotlib", "scipy", "sklearn"],
        default=None
    )
    
    if st.button("Convert to Code"):
        if input_text:
            with st.spinner("Converting to code... This may take a moment."):
                try:
                    output_code = generate_code(input_text, language.lower(), libraries)
                    st.header("Output: Code Implementation")
                    st.code(output_code, language=language.lower())
                    
                    # Add a download button for the generated code
                    st.download_button(
                        label="Download Code",
                        data=output_code,
                        file_name=f"generated_code.{language.lower()}",
                        mime="text/plain"
                    )
                except Exception as e:
                    logger.error(f"Error processing input: {str(e)}")
                    st.error("An error occurred while processing your input. Please try again.")
        else:
            st.warning("Please enter some text or upload a PDF file.")

if __name__ == "__main__":
    main()
