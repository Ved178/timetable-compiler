import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project modules
from src.image_processor import load_image, enhance_image_for_ocr
from src.table_detector import detect_table_structure
from src.ocr_extractor import OCRExtractor
from src.llm_corrector import LLMCorrector
from src.timetable_merger import TimetableMerger
from src.utils import (
    save_uploaded_file, 
    generate_output_filename, 
    clean_temp_files,
    validate_api_key,
    format_dataframe_for_display
)
from config import SUPPORTED_LLM_PROVIDERS, DEFAULT_MODEL

# Page configuration
st.set_page_config(
    page_title="Timetable Compiler",
    layout="wide"
)

# Title and description
st.title("Timetable Compiler")
st.markdown("""
Upload multiple timetable images and compile them into one accurate DataFrame using OCR and LLM correction.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# LLM Provider selection
provider = st.sidebar.selectbox(
    "LLM Provider",
    options=list(SUPPORTED_LLM_PROVIDERS.keys()),
    format_func=lambda x: SUPPORTED_LLM_PROVIDERS[x]
)

# API Key input
api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help=f"Enter your {SUPPORTED_LLM_PROVIDERS[provider]} API key"
)

# Model selection
model = st.sidebar.text_input(
    "Model (optional)",
    value=DEFAULT_MODEL.get(provider, ""),
    help="Leave blank to use default model"
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    scale = st.slider("Table Detection Scale", 10, 30, 15, 
                     help="Lower values detect more lines")
    joint_tol = st.slider("Joint Tolerance", 6, 20, 12,
                         help="Clustering tolerance in pixels")
    use_gpu = st.checkbox("Use GPU for OCR", value=False,
                         help="Enable if CUDA is available")
    batch_size = st.slider("LLM Batch Size", 5, 20, 10,
                          help="Number of entries to correct per API call")

# Main content
tab1, tab2 = st.tabs(["Upload & Process", "Instructions"])

with tab1:
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Timetable Images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload one or more timetable images"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        
        # Display uploaded images
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 3]:
                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
    
    # Process button
    if st.button("Process Timetables", type="primary", disabled=not uploaded_files):
        if not api_key:
            st.error("Please enter your API key in the sidebar")
        elif not validate_api_key(provider, api_key):
            st.error(f"Invalid API key format for {provider}")
        else:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Clean previous temp files
                clean_temp_files()
                
                all_dataframes = []
                all_labels = []
                
                # Process each uploaded file
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    progress = (idx) / (len(uploaded_files) * 4)
                    progress_bar.progress(progress)
                    
                    # Save uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Step 1: Load and enhance image
                    status_text.text(f"Enhancing {uploaded_file.name}...")
                    img = load_image(str(file_path))
                    enhanced_img = enhance_image_for_ocr(img)
                    progress_bar.progress((idx + 0.25) / (len(uploaded_files)))
                    
                    # Step 2: Detect table structure
                    status_text.text(f"Detecting table in {uploaded_file.name}...")
                    cells, table_roi = detect_table_structure(
                        enhanced_img, 
                        scale=scale, 
                        joint_tol=joint_tol
                    )
                    progress_bar.progress((idx + 0.5) / (len(uploaded_files)))
                    
                    # Step 3: Extract text with OCR
                    status_text.text(f"Extracting text from {uploaded_file.name}...")
                    ocr_extractor = OCRExtractor(gpu=use_gpu)
                    df, table_text = ocr_extractor.extract_full_table(cells)
                    progress_bar.progress((idx + 0.75) / (len(uploaded_files)))
                    
                    # Step 4: Correct with LLM
                    status_text.text(f"Correcting with LLM: {uploaded_file.name}...")
                    llm_corrector = LLMCorrector(
                        provider=provider,
                        model=model if model else None,
                        api_key=api_key
                    )
                    corrected_df = llm_corrector.batch_correct_dataframe(
                        df, 
                        batch_size=batch_size
                    )
                    progress_bar.progress((idx + 1) / (len(uploaded_files)))
                    
                    all_dataframes.append(corrected_df)
                    all_labels.append(uploaded_file.name)
                
                # Step 5: Merge all timetables
                status_text.text("Merging timetables...")
                merger = TimetableMerger()
                merged_df = merger.merge_timetables(all_dataframes, all_labels)
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Display results
                st.success("Timetables processed successfully!")
                
                # Show individual results
                st.subheader("Individual Timetables")
                for label, df in zip(all_labels, all_dataframes):
                    with st.expander(f"View {label}"):
                        st.dataframe(
                            format_dataframe_for_display(df),
                            use_container_width=True
                        )
                
                # Show merged result
                st.subheader("Merged Timetable")
                st.dataframe(merged_df, use_container_width=True)
                
                # Export options
                st.subheader("Export")
                col1, col2 = st.columns(2)
                
                with col1:
                    export_format = st.radio("Format", ["xlsx", "csv"])
                
                with col2:
                    if st.button("Download Merged Timetable"):
                        output_path = generate_output_filename(
                            prefix="merged_timetable",
                            format=export_format
                        )
                        merger.export_merged_timetable(
                            merged_df, 
                            output_path, 
                            format=export_format
                        )
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label=f"Download {export_format.upper()}",
                                data=f,
                                file_name=output_path.name,
                                mime=f"application/{export_format}"
                            )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

with tab2:
    st.markdown("""
    ## How to Use
    
    ### 1. Setup
    - Choose your LLM provider (OpenAI, Anthropic, or Google)
    - Enter your API key in the sidebar
    - Optionally, adjust advanced settings
    
    ### 2. Upload
    - Click "Browse files" to upload timetable images
    - You can upload multiple files at once
    - Supported formats: JPG, PNG
    
    ### 3. Process
    - Click "Process Timetables" to start
    - The app will:
      - Enhance images for better OCR
      - Detect table structures
      - Extract text using EasyOCR
      - Correct errors using LLM
      - Merge all timetables
    
    ### 4. Export
    - Review individual and merged timetables
    - Download as Excel (.xlsx) or CSV
    
    ## Tips for Best Results
    
    - **Image Quality**: Use high-resolution, well-lit images
    - **Table Structure**: Ensure tables have clear borders
    - **Orientation**: Upload images in correct orientation
    - **Class Names**: Edit `prompts/system_prompt.txt` to include your specific class names
    
    ## Advanced Settings
    
    - **Table Detection Scale**: Lower values (10-15) detect more lines, higher values (20-30) are more selective
    - **Joint Tolerance**: Controls how nearby points are clustered into grid intersections
    - **GPU**: Enable for faster OCR if you have CUDA-capable GPU
    - **Batch Size**: Higher values make fewer API calls but may hit token limits
    
    ## Customization
    
    To add your own class names:
    1. Edit `prompts/system_prompt.txt`
    2. Add your classes to the "Valid Class Names" section
    3. Restart the app
    
    ## Supported LLM Providers
    
    - **OpenAI**: GPT-4, GPT-3.5-turbo
    - **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
    - **Google**: Gemini Pro
    """)

