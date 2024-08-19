import streamlit as st
import google.generativeai as genai
import os
import docx2txt
import PyPDF2 as pdf
import re
from googlesearch import search
import numpy as np
import zipfile
import tempfile

# Set up your Google API key
os.environ['GOOGLE_API_KEY'] = "AIzaSyCfGPIrdQ4ratJzojK81RyDluE22BiuZoc"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.05,
    "top_p": 0.95,
    "top_k": 10,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

def generate_response_from_gemini(input_text):
    try:
        llm = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        output = llm.generate_content(input_text)
        return output.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def extract_text_from_pdf_file(uploaded_file):
    try:
        pdf_reader = pdf.PdfReader(uploaded_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += str(page.extract_text())
        return text_content
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx_file(uploaded_file):
    try:
        return docx2txt.process(uploaded_file)
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_job_title_and_location(job_description):
    try:
        job_title_match = re.search(r"Job Title:\s*(.*)", job_description)
        location_match = re.search(r"Location:\s*(.*)", job_description)
        job_title = job_title_match.group(1).strip() if job_title_match else "Not found"
        location = location_match.group(1).strip() if location_match else "Not found"
        return job_title, location
    except Exception as e:
        st.error(f"Error extracting job title and location: {e}")
        return "Not found", "Not found"

input_prompt_template = """
As an experienced Applicant Tracking System (ATS) analyst, with profound knowledge in technology, software engineering, data science, and big data engineering, your role involves evaluating resumes against job descriptions. Recognizing the competitive job market, provide top-notch assistance for resume improvement. Your goal is to analyze the resume against the given job description, assign a percentage match based on key criteria, and pinpoint missing keywords accurately. resume:{text} description:{job_description} I want the response in one single string having the structure {{"Job Description Match":"%","Missing Keywords":"","Candidate Summary":"","Experience":""}}
"""

def search_profiles_linkedin(job_title, location):
    query = f"{job_title} profiles in {location} site:linkedin.com"
    try:
        results = search(query, tld="com", lang="en", num=15, stop=15, pause=1)
        return list(results)
    except Exception as e:
        st.error(f"Error searching LinkedIn profiles: {e}")
        return []

def scrape_remove_url(results):
    unwanted_patterns = [
        'https://in.linkedin.com/jobs/',
        'https://www.linkedin.com/posts/'
    ]
    filtered_results = [r for r in results if not any(r.startswith(pattern) for pattern in unwanted_patterns)]
    return filtered_results

def get_user_feedback(results):
    relevant_results = []
    for url in results:
        user_input = st.text_input(f"Is this URL relevant and correct? (yes/no): {url}", key=url)
        if user_input.lower() == 'yes':
            relevant_results.append(url)
    accuracy = (len(relevant_results) / len(results)) * 100 if results else 0
    return relevant_results, accuracy

def evaluate_resume(resume_text, job_description):
    response_text = generate_response_from_gemini(input_prompt_template.format(text=resume_text, job_description=job_description))
    if response_text:
        match_percentage_str = response_text.split('"Job Description Match":"')[1].split('"')[0]
        match_percentage = float(match_percentage_str.rstrip('%'))
        missing_keywords_str = response_text.split('"Missing Keywords":"')[1].split('"')[0]
        candidate_summary_str = response_text.split('"Candidate Summary":"')[1].split('"')[0]
        experience_str = response_text.split('"Experience":"')[1].split('"')[0]
        return match_percentage, missing_keywords_str, candidate_summary_str, experience_str
    else:
        return 0, "Error", "Error", "Error"

# Streamlit app
st.title("Intelligent ATS and LinkedIn Profile Search")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["ATS Evaluation", "LinkedIn Profile Search", "LinkedIn Candidate Search"])

if page == "ATS Evaluation":
    st.header("ATS Evaluation")
    
    jd_file = st.file_uploader("Upload the Job Description File", type=["pdf", "docx"], help="Please upload PDF or DOCX file")
    uploaded_zip = st.file_uploader("Upload the Resume Folder (ZIP)", type=["zip"], help="Upload a ZIP folder containing PDF or DOCX resumes")

    job_description = ""
    if jd_file:
        if jd_file.type == "application/pdf":
            job_description = extract_text_from_pdf_file(jd_file)
        elif jd_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            job_description = extract_text_from_docx_file(jd_file)
        
        job_title, location = extract_job_title_and_location(job_description)
        st.write(f"**Job Title:** {job_title}")
        st.write(f"**Location:** {location}")

    submit_button = st.button("Submit")

    if submit_button:
        if uploaded_zip and job_description:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                resume_files = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir) if file.endswith(('.pdf', '.docx'))]

                no_match = True
                for resume_file in resume_files:
                    if resume_file.endswith(".pdf"):
                        with open(resume_file, "rb") as file:
                            resume_text = extract_text_from_pdf_file(file)
                    elif resume_file.endswith(".docx"):
                        resume_text = extract_text_from_docx_file(resume_file)

                    num_runs = 3
                    match_percentages = []
                    for _ in range(num_runs):
                        match_percentage, missing_keywords_str, candidate_summary_str, experience_str = evaluate_resume(resume_text, job_description)
                        match_percentages.append(match_percentage)

                    avg_match_percentage = np.mean(match_percentages)

                    st.subheader(f"ATS Evaluation Result for {os.path.basename(resume_file)}:")
                    st.write(f'Match to Job Description: {avg_match_percentage}%')
                    st.write("Keywords Missing: ", missing_keywords_str)
                    st.write("Summary of Resume of Candidate: ")
                    st.write(candidate_summary_str)
                    st.write("Experience: ", experience_str)

                    if avg_match_percentage >= 75:
                        st.text("Move forward with hiring")
                        no_match = False
                    else:
                        st.text("Not a Match")

                if no_match:
                    st.session_state["show_linkedin_profiles"] = True
                    st.session_state["job_title"] = job_title
                    st.session_state["location"] = location
        else:
            st.warning("Please upload both the resume folder and job description file.")

elif page == "LinkedIn Profile Search":
    if "show_linkedin_profiles" not in st.session_state:
        st.session_state["show_linkedin_profiles"] = False

    if st.session_state["show_linkedin_profiles"]:
        if 'job_title' not in st.session_state or 'location' not in st.session_state:
            st.session_state['job_title'] = ''
            st.session_state['location'] = ''

        job_title = st.session_state['job_title']
        location = st.session_state['location']

        st.write(f"**Job Title:** {job_title}")
        st.write(f"**Location:** {location}")

        if job_title and location:
            results_linkedin = search_profiles_linkedin(job_title, location)
            results_best_match = scrape_remove_url(results_linkedin)

            if results_best_match:
                st.subheader("LinkedIn Profiles:")
                relevant_results, accuracy = get_user_feedback(results_best_match)
                st.write(f"Relevance Accuracy: {accuracy}%")
                for url in relevant_results:
                    st.write(url)
            else:
                st.write("No LinkedIn profiles found.")

        if st.button("Search LinkedIn Profiles"):
            st.session_state["show_linkedin_profiles"] = False
    else:
        st.write("Please go to the ATS Evaluation page first.")

elif page == "LinkedIn Candidate Search":
    st.header("LinkedIn Candidate Search")
    
    job_title_input = st.text_input("Enter the Job Title", help="Enter the job title you want to search for")
    location_input = st.text_input("Enter the Location", help="Enter the location you want to search in")

    if st.button("Search LinkedIn Candidates"):
        if job_title_input and location_input:
            results_linkedin = search_profiles_linkedin(job_title_input, location_input)
            results_best_match = scrape_remove_url(results_linkedin)
            if results_best_match:
                st.subheader("LinkedIn Profiles:")
                relevant_results, accuracy = get_user_feedback(results_best_match)
                st.write(f"Relevance Accuracy: {accuracy}%")
                for url in relevant_results:
                    st.write(url)
            else:
                st.write("No LinkedIn profiles found.")
        else:
            st.warning("Please enter both a job title and location.")
