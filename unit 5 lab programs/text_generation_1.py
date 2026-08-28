import streamlit as st
import ollama

st.set_page_config(
    page_title="Local LLM Text Generator",
    page_icon="🤖"
)

st.title("🤖 Local LLM Text Generation")
st.write("Generate text using a locally running Large Language Model.")

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Write a short story about artificial intelligence..."
)

model = st.selectbox(
    "Select Local Model",
    ["llama3.2", "mistral", "phi3"]
)

if st.button("Generate Text"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating text..."):
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                generated_text = response["message"]["content"]

                st.subheader("Generated Text")
                st.write(generated_text)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info(
                    "Make sure Ollama is running and the selected model "
                    "has been downloaded."
                )