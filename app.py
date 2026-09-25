
import streamlit as st

from graph import research_graph


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")
st.write(
    "Explore research papers using RAG, FAISS, Groq and LangGraph."
)

st.sidebar.header("Research Assistant")

operation = st.sidebar.selectbox(
    "Choose operation",
    [
        "Question & Answer",
        "Explain Simply",
        "Key Findings"
    ]
)

question = st.text_area(
    "Enter your question:",
    placeholder="Ask something about the research paper...",
    height=100
)


if st.button("Generate Answer", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        if operation == "Explain Simply":
            graph_question = "Explain simply: " + question

        elif operation == "Key Findings":
            graph_question = "What are the key findings of the paper?"

        else:
            graph_question = question

        with st.spinner("Analyzing the research paper..."):

            result = research_graph.invoke({
                "question": graph_question,
                "operation": "",
                "answer": "",
                "sources": []
            })

        st.subheader("Answer")
        st.write(result["answer"])

        st.caption(
            f"Operation used: {result['operation']}"
        )

        if result.get("sources"):
            st.subheader("Sources")

            shown = set()

            for source in result["sources"]:
                key = (
                    source["source"],
                    source["page"]
                )

                if key not in shown:
                    shown.add(key)

                    st.write(
                        f"📄 {source['source']} — Page {source['page']}"
                    )


st.divider()

st.caption(
    "Powered by RAG • FAISS • Sentence Transformers • Groq • LangGraph"
)
