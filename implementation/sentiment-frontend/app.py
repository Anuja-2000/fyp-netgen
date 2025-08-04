import streamlit as st
import requests

st.title("Submit Your Review")

review = st.text_area("Your Review", placeholder="Enter your review here...")
user_type = st.selectbox("User Type", options=["solo", "couple", "family", "groups", "unkown"])
location_name = st.text_input("Location Name", placeholder="e.g., Arugam Bay")

if st.button("Submit"):
    data = {
        "review": review,
        "user_type": user_type,
        "location_name": location_name
    }
    try:
        response = requests.post("http://127.0.0.1:5000/predict_review", json=data)
        result = response.json()
        st.subheader("Analysis Results")
        st.write(f"**User Type:** {result.get('user_type', 'N/A')}")
        st.write(f"**Number of Sentences:** {result.get('num_sentences', 'N/A')}")
        for idx, sentence_info in enumerate(result.get("sentence_results", []), 1):
            st.markdown(f"---\n**Sentence {idx}:** {sentence_info.get('sentence', '')}")
            st.write(f"- **Safety Label:** {'Unsafe' if sentence_info.get('safety_label', 0) == 1 else 'Safe'}")
            aspects = sentence_info.get('aspects', {})
            if aspects:
                with st.expander("Show Aspects & Sentiments", expanded=False):
                    for aspect, sentiment in aspects.items():
                        st.markdown(
                            f"<div style='padding-left:20px;'>"
                            f"<span style='font-weight:bold;color:#4F8BF9;'>{aspect.capitalize()}</span>: "
                            f"<span style='color:#333;'>{sentiment}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.write("- **Aspects:** None")
    except Exception as e:
        st.error(f"Error: {e}")

        