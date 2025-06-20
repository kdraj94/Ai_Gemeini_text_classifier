import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import config_file

# --- Environment and API Key Setup ---
# It's recommended to set your Google API key as an environment variable.
# For Streamlit Cloud, you can set this in the app's secrets.
# For local development, you can create a .env file and load it,
# or set the environment variable in your system.
# Example for local development using a .env file (you'd need to install python-dotenv):
# from dotenv import load_dotenv
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# For simplicity in this example, we'll use Streamlit's secrets management,
# which is the best practice for deploying apps.
# When running locally, you can create a .streamlit/secrets.toml file.
try:
    # Attempt to get the API key from Streamlit's secrets
    GOOGLE_API_KEY = config_file.api_key
except (FileNotFoundError, KeyError):
    # Fallback for local development if secrets.toml is not used
    # In this case, it will look for an environment variable
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in your environment or Streamlit secrets.")
    st.stop()

# --- LangChain and Gemini Model Initialization ---
# Using "gemini-1.5-flash" as it's a fast and versatile model.
# The error "gemini-pro not found" suggests the model name needed an update.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# --- Prompt Template ---
# This template guides the language model to perform the classification task.
prompt_template = """
You are an expert in analyzing customer complaints.
Your task is to classify the following telecom customer complaint into one of these categories:
- Billing Issue
- Technical Support
- Product Feedback
- Product Quality Issue
- Shipping/Delivery Problem
- Billing/Payment Issue
- Customer Service Experience
- Feature Request/Suggestion
- General Inquiry/Other

Complaint:
"{complaint_text}"

Category:
"""

prompt = PromptTemplate.from_template(prompt_template)

# --- LangChain Expression Language (LCEL) Chain ---
# This is the modern way to chain components in LangChain.
# It pipes the prompt to the model and then to an output parser.
chain = prompt | llm | StrOutputParser()


# --- Streamlit User Interface ---
st.set_page_config(page_title="Customer Complaint Classifier", page_icon="🤖")

st.title("Customer Complaint Classifier")
st.write("Enter a customer complaint below and click 'Classify' to determine its category.")

# Text area for user input
user_input = st.text_area("Customer Complaint:", height=68)

# Button to trigger classification
if st.button("Classify"):
    if user_input:
        with st.spinner("Classifying..."):
            try:
                # Run the chain with the user's input
                result = chain.invoke({"complaint_text": user_input})
                st.success("Classification Result:")
                st.write(result.strip())
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a complaint to classify.")


