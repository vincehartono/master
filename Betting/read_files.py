import os
import PyPDF2

def combine_pdfs_to_text(folder_path, output_file):
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    pdf_files.sort()

    with open(output_file, "w", encoding="utf-8") as txt_file:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Reading: {pdf_file}")

            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num in range(len(reader.pages)):
                        text = reader.pages[page_num].extract_text()
                        if text:
                            txt_file.write(f"\n\n--- {pdf_file} | Page {page_num + 1} ---\n\n")
                            txt_file.write(text)
            except Exception as e:
                print(f"❌ Error reading {pdf_file}: {e}")

    print(f"\n✅ All PDFs combined into: {output_file}")

# Run for your folder
folder_path = r"C:\Dropbox\Vincent\Actuary\DMAC\ERM Case Study"
output_file = r"C:\Dropbox\Vincent\Actuary\DMAC\combined_text.txt"
combine_pdfs_to_text(folder_path, output_file)