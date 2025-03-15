import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
MODEL_PATH = "./model"  # Update if your model is in a different path

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")  # Use "cuda" if you have a GPU
    return tokenizer, model

tokenizer, model = load_model()
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Streamlit UI
st.title("🤖 AI Riddle Solver")
st.write("Enter a riddle, and the AI will try to solve it!")

user_input = st.text_area("Enter your riddle:", "")

if st.button("Solve Riddle"):
    if user_input.strip():
        response = generator(user_input + "\nAnswer:", max_new_tokens=50, num_return_sequences=1)
        st.subheader("AI's Answer:")
        st.write(response[0]['generated_text'])
    else:
        st.warning("Please enter a riddle first.")

