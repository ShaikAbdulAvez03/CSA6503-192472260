import streamlit as st
import ollama

st.set_page_config(
    page_title="Local LLM Text Summarizer",
    page_icon="📝"
)

st.title("📝 Local LLM Text Summarization")
st.write("Summarize text using a locally running Large Language Model.")

text = st.text_area(
    "Enter text to summarize:",
    height=300,
    placeholder="Paste a long article or paragraph here..."
)

model = st.selectbox(
    "Select Local Model",
    ["llama3.2", "mistral", "phi3"]
)

summary_length = st.selectbox(
    "Summary Length",
    ["Short", "Medium", "Detailed"]
)

if st.button("Summarize Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        if summary_length == "Short":
            instruction = (
                "Summarize the following text in 3 to 5 sentences."
            )
        elif summary_length == "Medium":
            instruction = (
                "Summarize the following text in one clear paragraph "
                "containing the important points."
            )
        else:
            instruction = (
                "Provide a detailed summary of the following text. "
                "Include all important facts and key points."
            )

        prompt = f"""
{instruction}

Do not add information that is not present in the original text.

Text:
{text}
"""

        with st.spinner("Generating summary..."):
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

                summary = response["message"]["content"]

                st.subheader("Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info(
                    "Make sure Ollama is running and the selected model "
                    "has been downloaded."
                )