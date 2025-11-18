from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_text(text_path):
    with open(text_path, 'r') as f:
        return f.read()

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

def answer_question(question, metadata_path, text_path):

    # Load all content
    text = load_text(text_path)
    metadata = load_metadata(metadata_path)

    combined_context = text
    for entry in metadata:
        if "caption" in entry:
            combined_context += "\n" + entry["caption"]

    # Embeddings
    embeddings = model.encode([combined_context, question])
    score = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Response
    answer = (
        f"**Relevant Content Found (score: {score:.3f}):**\n\n"
        f"{combined_context[:1200]}...\n\n"
        "*(Context truncated for display)*"
    )
    return answer

