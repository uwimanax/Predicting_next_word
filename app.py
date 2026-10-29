import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Group 1 — Transformer Demo", page_icon="🤖", layout="wide")

GPT2_NAME = "gpt2"
QWEN_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# The notebook's small document set, used for a lightweight RAG demonstration.
DOCUMENTS = [
    {
        "source": "HEC Rwanda, Contact List for all Institutions (2025)",
        "text": (
            "Public higher learning institutions in Rwanda accredited by the Higher Education Council (HEC): "
            "University of Rwanda (UR), Rwanda Polytechnic (RP), and the Institute of Legal Practice and Development (ILPD)."
        ),
    },
    {
        "source": "HEC Rwanda, Contact List for all Institutions (2025)",
        "text": (
            "Private higher learning institutions in Rwanda include Independent University of Kigali (ULK), "
            "Institute of Lay Adventists of Kigali (INILAK), Catholic University of Rwanda (CUR), "
            "Adventist University of Central Africa (AUCA), Institut Catholique de Kabgayi (ICK), INES Ruhengeri, "
            "Institut Polytechnique de Byumba (IPB), Rwanda Tourism University College (RTUC), "
            "Protestant Institute of Arts and Social Sciences (PIASS), Mount Kigali University (MKU), "
            "Kibogora Polytechnic (KP), Carnegie Mellon University Africa (CMUA), University of Kigali (UoK), "
            "East African University Rwanda (EAUR), and University of Global Health Equity (UGHE)."
        ),
    },
    {
        "source": "Group 1 notebook — model comparison",
        "text": (
            "GPT-2 small has about 124 million parameters and is primarily trained on English text. "
            "Qwen3-4B-Instruct-2507 has about 4 billion parameters and is instruction-tuned. "
            "The notebook compares tokenization, next-token prediction, attention, and text generation."
        ),
    },
]

@st.cache_resource(show_spinner="Loading model… first launch can take several minutes.")
def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": dtype}
    if model_name == QWEN_NAME and device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not (model_name == QWEN_NAME and device == "cuda"):
        model = model.to(device)
    model.eval()
    return tokenizer, model, device

def generate_answer(tokenizer, model, device, prompt, max_new_tokens, chat_mode):
    if chat_mode:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant for Group 1's transformer project. "
                    "Answer clearly and helpfully. You may answer in English or Kinyarwanda."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_ids = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

def retrieve_docs(question, k=2):
    # Lightweight keyword-based retrieval avoids downloading a second large embedding model.
    terms = {word.lower().strip(".,?!:;()") for word in question.split() if len(word) > 2}
    scored = []
    for doc in DOCUMENTS:
        text_terms = set(doc["text"].lower().split())
        score = len(terms & text_terms)
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored[:k] if score > 0]

st.title("🤖 Group 1: Inside GPT-2 and Qwen3")
st.caption("A Streamlit version of the notebook's text-generation and RAG demonstrations.")

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Choose a model",
        ["GPT-2 small (lighter)", "Qwen3-4B-Instruct (large)"],
    )
    use_rag = st.checkbox("Use notebook documents (RAG demo)", value=False)
    max_new_tokens = st.slider("Maximum new tokens", min_value=20, max_value=250, value=120, step=10)
    st.divider()
    st.write("**Device detected:**", "CUDA GPU" if torch.cuda.is_available() else "CPU")
    if selected_model.startswith("Qwen3"):
        st.warning("Qwen3-4B is large. A GPU with sufficient VRAM is recommended; CPU generation may be very slow.")
    st.caption("The RAG demo uses the small set of documents included in this app. It is not a complete university directory.")

tab_chat, tab_next, tab_info = st.tabs(["Chat", "Continue a prompt", "About"])

with tab_chat:
    st.subheader("Ask the model")
    question = st.text_area(
        "Enter your question in English or Kinyarwanda",
        placeholder="For example: Explain attention in a transformer.",
        height=100,
        key="chat_question",
    )
    if st.button("Generate answer", type="primary", key="chat_submit"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            model_name = GPT2_NAME if selected_model.startswith("GPT-2") else QWEN_NAME
            try:
                with st.spinner("Loading the model and generating an answer…"):
                    tokenizer, model, device = load_model(model_name)
                    prompt = question.strip()
                    sources = []
                    if use_rag:
                        docs = retrieve_docs(prompt)
                        if docs:
                            context = "\n\n".join(f"[Document] {d['text']}" for d in docs)
                            prompt = (
                                "Answer using only the context below. If the answer is not in the context, "
                                "say that the provided documents do not contain the answer. "
                                "Answer in the same language as the question.\n\n"
                                f"Context:\n{context}\n\nQuestion: {question.strip()}"
                            )
                            sources = sorted({d["source"] for d in docs})
                    answer = generate_answer(
                        tokenizer, model, device, prompt, max_new_tokens,
                        chat_mode=selected_model.startswith("Qwen3"),
                    )
                st.markdown("**Answer**")
                st.write(answer if answer else "The model returned an empty answer. Try a different prompt.")
                if use_rag:
                    st.markdown("**Retrieved sources**")
                    if sources:
                        for source in sources:
                            st.write(f"- {source}")
                    else:
                        st.write("No matching notebook document was found; the answer may not be grounded in the supplied documents.")
            except Exception as exc:
                st.error("The model could not be loaded or run.")
                st.exception(exc)
                st.info("Check the installation steps in README.txt. If the error mentions memory, try GPT-2 first.")

with tab_next:
    st.subheader("Continue a sentence")
    prompt_text = st.text_area(
        "Enter the beginning of a sentence",
        value="The people of Kicukiro say they have not had water for three",
        height=90,
        key="next_prompt",
    )
    if st.button("Continue text", key="continue_submit"):
        if not prompt_text.strip():
            st.warning("Please enter a prompt.")
        else:
            model_name = GPT2_NAME if selected_model.startswith("GPT-2") else QWEN_NAME
            try:
                with st.spinner("Generating continuation…"):
                    tokenizer, model, device = load_model(model_name)
                    continuation = generate_answer(
                        tokenizer, model, device, prompt_text.strip(), max_new_tokens,
                        chat_mode=False,
                    )
                st.markdown("**Generated continuation**")
                st.write(prompt_text.strip() + continuation)
            except Exception as exc:
                st.error("The model could not be loaded or run.")
                st.exception(exc)

with tab_info:
    st.subheader("What this application demonstrates")
    st.markdown(
        """
        - **GPT-2 small:** text continuation and next-token-style generation.
        - **Qwen3-4B-Instruct:** instruction-following chat.
        - **RAG demonstration:** retrieves relevant text from a small set of documents adapted from the notebook and supplies it as context.
        - **English and Kinyarwanda:** you can try prompts in either language; output quality can vary by model.

        The notebook also contains attention visualizations and a Sunflower model experiment. This first Streamlit app focuses on a practical interactive demo.
        """
    )
