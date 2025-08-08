# AI-Powered Job Recommendation System

## [cite_start]1. Project Description [cite: 19]
This project is an AI-powered web application designed to help students find suitable job and internship opportunities. It leverages a Retrieval-Augmented Generation (RAG) model to match a user's skills, experience, and preferences against a database of job postings and provide personalized recommendations with explanations.

## [cite_start]2. Features [cite: 20]
* [cite_start]**Manual Data Entry**: Users can manually enter their skills, experience, and job preferences. [cite: 7]
* [cite_start]**Resume Upload**: Users can upload their resume in PDF format for automated analysis. [cite: 7]
* [cite_start]**Personalized Recommendations**: The system uses LangChain and a Groq-powered LLM to generate smart recommendations. [cite: 9]
* [cite_start]**Clear Explanations**: Each recommendation comes with a short explanation of why the job is a good fit. [cite: 9]
* [cite_start]**Interactive UI**: A simple and user-friendly web interface built with Streamlit. [cite: 10]

## 3. Tech Stack
| Component | Technology |
|---|---|
| Programming | Python |
| LLM Integration | LangChain + Groq API |
| Vector Store | FAISS |
| UI | Streamlit |
| Deployment | GitHub |

## [cite_start]4. How to Run [cite: 21]
Follow these steps to run the application locally:

**Prerequisites:**
* Python 3.8+ installed
* A Groq API Key

**Steps:**
1.  Clone the repository:
    ```bash
    git clone <your-github-repo-link>
    ```
2.  Navigate to the project directory:
    ```bash
    cd GenAI-Job-Project
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## [cite_start]5. Sample Input/Output [cite: 22]
**Sample Input (Manual Entry):**
* **Skills:** `Python, Pandas, Matplotlib, basic machine learning`
* **Experience:** `College projects in data analysis`
* **Preferences:** `Data Science Internship, Work from Home`

**Sample Output:**
> **Here are your personalized recommendations:**
>
> I would recommend the **Data Science Intern** position at **Data Insights Co.**
>
> *Why it's a good match:* This is an excellent fit because it's an internship that directly requires the Python, Pandas, and machine learning skills you mentioned. Furthermore, it's a "Work from Home" role, which perfectly aligns with your preferences.

## [cite_start]6. Screenshots [cite: 23]
*(Here, insert screenshots of your running Streamlit application. Show the home screen and a screen with results.)*

![App Screenshot 1](link_to_your_screenshot_1.png)
![App Screenshot 2](link_to_your_screenshot_2.png)