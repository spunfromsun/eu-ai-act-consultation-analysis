

def main() -> None:
    # %% Cell 1
    import sys
    print(sys.executable)

    # %% Cell 2
    import pandas as pd

    # %% Cell 3
    # NOTE: Jupyter magic: %pip install --upgrade pip
    # NOTE: Jupyter magic: %pip install pandas numpy
    # useful extras (optional)
    # NOTE: Jupyter magic: %pip install matplotlib seaborn    # plotting
    # NOTE: Jupyter magic: %pip install chardet               # encoding detection if read_csv fails
    # NOTE: Jupyter magic: %pip install openpyxl pyarrow      # excel / parquet support
    # NOTE: Jupyter magic: %pip install tqdm

    # %% Cell 4
    import os
    from pathlib import Path

    # %% Cell 5
    # Check current working directory
    print("Current working directory:")
    print(os.getcwd())

    # List files in current directory
    print("\nFiles in current directory:")
    for f in os.listdir('.'):
        print(f)

    # %% Cell 6
    import pandas as pd

    # %% Cell 7
    import chardet

    # Detect encoding
    with open("contributions-AIConsult2020.csv", 'rb') as f:
        raw_data = f.read()
        encoding = chardet.detect(raw_data)
        print(f"Detected encoding: {encoding}")

    # %% Cell 8
    # Try different encodings
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'windows-1252']

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv("contributions-AIConsult2020.csv", encoding=encoding)
            print(f"Success with encoding: {encoding}")
            print(f"Shape: {df.shape}")
            break
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
        except Exception as e:
            print(f"Other error with {encoding}: {e}")

    # %% Cell 9
    # Option 1: Let pandas handle inconsistent columns
    df = pd.read_csv("contributions-AIConsult2020.csv", 
                     encoding='windows-1252',
                     on_bad_lines='skip')  # skip problematic lines

    print(f"Loaded {len(df)} rows")
    df.head()

    # %% Cell 10
    # Read with semicolon separator
    df = pd.read_csv("contributions-AIConsult2020.csv", 
                     encoding='windows-1252',
                     sep=';',  # Use semicolon separator
                     on_bad_lines='skip')

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.shape[1]}")
    print(f"Column names: {list(df.columns)}")
    df.head()

    # %% Cell 11
    df["User type"].value_counts()

    # %% Cell 12
    # Filter for NGO and Trade Union only
    filtered_df = df[df["User type"].isin(["NGO (Non-governmental organisation)", "Trade Union"])]

    print(f"Filtered dataframe: {len(filtered_df)} rows")
    print(filtered_df["User type"].value_counts())

    # %% Cell 13
    filtered_df.head()

    # %% Cell 14
    # Save filtered df to excel
    with pd.ExcelWriter("Phase1_SurveyAnalysis_20251214.xlsx", engine='openpyxl') as writer:
        filtered_df.to_excel(writer, sheet_name='NGOs_TUs', index=False)

    # %% Cell 15
    import re
    from pathlib import Path

    # Path to your attachments folder
    attachments_path = Path("attachments-AIConsult2020")

    # Check if folder exists
    if not attachments_path.exists():
        print(f"Folder not found: {attachments_path}")
    else:
        print(f"Found folder: {attachments_path}")
        print(f"Number of files: {len(list(attachments_path.glob('*.pdf')))}")

    # %% Cell 16
    # Extract reference numbers and filenames
    pdf_data = []

    for pdf_file in attachments_path.glob("*.pdf"):
        filename = pdf_file.name
        
        # Extract reference number (first 7 characters: F + 6 numbers)
        ref_match = re.match(r'^(F\d{6})', filename)
        if ref_match:
            reference_number = ref_match.group(1)
            
            # Extract title (remove reference and extension, clean up)
            title_part = filename.replace(reference_number + "-", "").replace(".pdf", "")
            # Clean up title (remove dates, common patterns)
            title_clean = re.sub(r'_\d{8}_', '_', title_part)  # remove date pattern
            title_clean = title_clean.replace('_', ' ').strip()
            
            pdf_data.append({
                'reference_number': reference_number,
                'filename': filename,
                'title': title_clean,
                'file_path': str(pdf_file)
            })

    # Create DataFrame
    pdf_df = pd.DataFrame(pdf_data)
    print(f"Extracted {len(pdf_df)} PDF references")
    pdf_df.head()

    # %% Cell 17
    # Check for duplicates and summary
    print(f"Unique references: {pdf_df['reference_number'].nunique()}")
    print(f"Total files: {len(pdf_df)}")

    # Show reference numbers
    print("\nFirst 10 reference numbers:")
    print(pdf_df['reference_number'].head(10).tolist())

    # %% Cell 18
    # Sort by reference number frequency (most frequent first), then by reference number
    pdf_df_sorted = pdf_df.assign(
        ref_count=pdf_df['reference_number'].map(pdf_df['reference_number'].value_counts())
    ).sort_values(['ref_count', 'reference_number'], ascending=[False, True])

    print(f"Sorted dataframe with {len(pdf_df_sorted)} files")
    print(f"Most frequent reference numbers:")
    print(pdf_df_sorted['reference_number'].value_counts().head(10))

    # Show the sorted dataframe
    pdf_df_sorted.head(20)

    # %% Cell 19
    pdf_df_sorted_filtered = pdf_df_sorted[pdf_df_sorted['reference_number'].isin(df['Reference'])]

    pdf_df_sorted_filtered_enhanced = pdf_df_sorted_filtered.merge(
        df[['Reference', 'User type', 'Organisation name']],
        left_on='reference_number',
        right_on='Reference',
        how='left'
    ).drop(columns=['Reference'])   

    # %% Cell 20
    pdf_df_sorted_filtered_enhanced.head()

    # %% Cell 21
    # Get unique reference numbers from filtered_df
    filtered_references = filtered_df['Reference'].unique()
    print(f"Number of unique references in filtered_df: {len(filtered_references)}")

    # %% Cell 22
    # Filter PDF dataframe and merge with CSV data to get User type and Organisation name
    pdf_filtered = pdf_df_sorted[pdf_df_sorted['reference_number'].isin(filtered_references)]

    # Merge with filtered_df to get User type and Organisation name
    pdf_filtered_enhanced = pdf_filtered.merge(
        filtered_df[['Reference', 'User type', 'Organisation name']], 
        left_on='reference_number', 
        right_on='Reference', 
        how='left'
    )

    # Drop the duplicate Reference column
    pdf_filtered_enhanced = pdf_filtered_enhanced.drop('Reference', axis=1)

    print(f"Enhanced filtered dataframe: {len(pdf_filtered_enhanced)} rows")
    print(f"New columns: {list(pdf_filtered_enhanced.columns)}")
    pdf_filtered_enhanced.head()

    # %% Cell 23
    # Filter PDF dataframe to only include references from filtered_df
    pdf_filtered = pdf_df_sorted[pdf_df_sorted['reference_number'].isin(filtered_references)]

    print(f"Original PDF files: {len(pdf_df_sorted)}")
    print(f"Filtered PDF files: {len(pdf_filtered)}")
    print(f"Matching references: {pdf_filtered['reference_number'].nunique()}")

    pdf_filtered.head()

    # %% Cell 24
    # Save both dataframes to different sheets in the same Excel file
    with pd.ExcelWriter("Phase1_Index_v2_20251120.xlsx", engine='openpyxl') as writer:
        pdf_filtered_enhanced.to_excel(writer, sheet_name='NGO_TradeUnion_PDFs', index=False)
        pdf_df_sorted_filtered_enhanced.to_excel(writer, sheet_name='All_PDFs', index=False)

    print(f"Excel file saved with:")
    print(f"- Sheet 1 'NGO_TradeUnion_PDFs': {len(pdf_filtered)} rows")
    print(f"- Sheet 2 'All_PDFs': {len(pdf_df_sorted)} rows")

    # %% Cell 25
    pdf_filtered_enhanced

    # %% Cell 26
    import shutil
    from pathlib import Path

    # Create NGO and Trade Union subfolders
    ngo_folder = attachments_path / "NGO"
    trade_union_folder = attachments_path / "Trade_Union"

    ngo_folder.mkdir(exist_ok=True)
    trade_union_folder.mkdir(exist_ok=True)

    print(f"Created folders: {ngo_folder} and {trade_union_folder}")

    # %% Cell 27
    # Move files based on user type
    moved_files = {'NGO': 0, 'Trade Union': 0, 'Not found': 0}

    for _, row in pdf_filtered_enhanced.iterrows():
        source_file = Path(row['file_path'])
        user_type = row['User type']
        
        if source_file.exists():
            if user_type == "NGO (Non-governmental organisation)":
                dest_file = ngo_folder / source_file.name
                moved_files['NGO'] += 1
            elif user_type == "Trade Union":
                dest_file = trade_union_folder / source_file.name
                moved_files['Trade Union'] += 1
            else:
                continue
                
            # Move the file
            shutil.move(str(source_file), str(dest_file))
            print(f"Moved: {source_file.name} -> {dest_file.parent.name}/")
        else:
            moved_files['Not found'] += 1

    print(f"\nSummary:")
    print(f"NGO files moved: {moved_files['NGO']}")
    print(f"Trade Union files moved: {moved_files['Trade Union']}")
    print(f"Files not found: {moved_files['Not found']}")



if __name__ == "__main__":
    main()

