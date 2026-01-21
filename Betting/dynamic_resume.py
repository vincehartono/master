import requests
from bs4 import BeautifulSoup
from docx import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import string

# ----------- Job scraping and keyword extraction -----------

def scrape_job_description(url):
    print("🔗 Scraping job description...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = " ".join(el.get_text() for el in soup.find_all(['p', 'li', 'span']) if el.get_text())
    return text

def extract_keywords(text, top_n=30):
    print("🔍 Extracting keywords...")

    nltk.download('punkt')
    nltk.download('stopwords')

    # 🧠 Bypass the internal `sent_tokenize()` call using `preserve_line=True`
    tokens = word_tokenize(text.lower(), language='english', preserve_line=True)
    words = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    word_freq = Counter(words)
    return set([word for word, _ in word_freq.most_common(top_n)])
# ----------- Resume filtering -----------

def read_resume(docx_path):
    print("📄 Reading resume...")
    doc = Document(docx_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def filter_resume(resume_lines, keywords):
    print("⚙️ Matching resume content...")
    matched = []
    for line in resume_lines:
        words = set(word.strip(string.punctuation).lower() for word in word_tokenize(line))
        if words & keywords:
            matched.append(line)
    return matched

# ----------- PDF writing -----------

def export_to_pdf(lines, filename="tailored_resume.pdf"):
    print(f"📝 Writing tailored resume to {filename}")
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = []

    for line in lines:
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 12))

    doc.build(content)
    print("✅ Done.")

# ----------- Main -----------

def main():
    print("📌 Resume Tailoring Tool")
    job_url = input("Paste the job description URL: ").strip()

    job_text = scrape_job_description(job_url)
    job_keywords = extract_keywords(job_text)

    resume_lines = read_resume(r"C:\Dropbox\Vincent\Resume\202505_Vince_Resume.docx")
    matched_lines = filter_resume(resume_lines, job_keywords)

    if matched_lines:
        export_to_pdf(matched_lines)
    else:
        print("⚠️ No matching resume lines found.")

if __name__ == "__main__":
    main()