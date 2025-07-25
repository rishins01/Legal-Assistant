from flask import Flask, render_template, request, session
from sentence_transformers import SentenceTransformer
import chromadb
import re
import os
from datetime import datetime
from markdown import markdown

from llms.groq_llm import GroqLLM
from langchain_core.prompts import PromptTemplate

# Set Groq API key
groq_api_key = os.getenv("")

app = Flask(__name__)
app.secret_key = ''

# Load model and connect to ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bns_laws")

# Initialize LangChain Groq LLM
groq_llm = GroqLLM(groq_api_key=groq_api_key)

def rephrase_with_llm(query):
    try:
        rephrase_prompt = f"""Rephrase the following user question about Indian laws into a more clear and concise legal query while preserving its meaning. Only return the improved query, no extra text.

Original Query:
{query}"""
        return groq_llm.invoke(rephrase_prompt).strip()
    except Exception as e:
        print(f"Query rephrase error: {e}")
        return query  # fallback to original if rephrasing fails




def generate_response_with_groq(query, context_docs, chat_history=[]):
    context_text = "\n\n".join(context_docs)

    history_context = ""
    for turn in chat_history[-5:]:
        response_summary = turn['response'][0].get('section_summary', '')
        if response_summary and response_summary != "❌ No matches found at all.":
            history_context += f"User: {turn['user_input']}\nAssistant: {response_summary}\n"

    prompt_template = PromptTemplate.from_template(
        """You are a helpful legal assistant.
Use the following conversation history and BNS legal context to answer the user's question.
If the context is not sufficient, say so. BNS is the new legal code and is said to replace Indian Penal Code or IPC
so do not mention anything like IPC or Indian Penal Code in your response.

Conversation History:
{history}

Legal Context:
{context}

Query:
{question}
"""
    )

    prompt = prompt_template.format(
        history=history_context.strip(),
        context=context_text,
        question=query
    )

    try:
        return groq_llm.invoke(prompt)
    except Exception as e:
        return f"❌ Groq LangChain API Error: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def home():
    result = []
    user_input = ""

    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        if request.form.get('clear'):
            session['chat_history'] = []
            return render_template('index.html', result=[], user_input="", chat_history=[])

        original_query = request.form['user_input']
        user_input = original_query.strip()

        if not user_input:
            result = [{
                "law_type": "Input Error",
                "section_summary": "⚠️ Please enter a query in the text box.",
                "source_url": "#",
                "match_type": "Error"
            }]
            return render_template('index.html', result=result, user_input=user_input, chat_history=session['chat_history'])

        # Do not rephrase yet, keep original query for now
        normalized_query = re.sub(
            r'till\s+section\s*(\d+)\s*from\s+(?:section\s*)?(\d+)',
            r'from section \2 to section \1',
            user_input,
            flags=re.IGNORECASE
        )

        MIN_SECTION = 1
        MAX_SECTION = 358
        extracted_sections = set()
        invalid_sections = set()

        # Parse from original query (before rephrasing)
        combined_range_matches = re.findall(
            r'(?:from\s+section\s*(\d+)\s*(?:to|till)\s*section?\s*(\d+))|(?:till\s+section\s*(\d+)\s*from\s+section\s*(\d+))',
            user_input,
            re.IGNORECASE
        )
        for m in combined_range_matches:
            nums = [n for n in m if n]
            if len(nums) == 2:
                start, end = int(nums[0]), int(nums[1])
                if start > end:
                    start, end = end, start
                for i in range(start, end + 1):
                    if MIN_SECTION <= i <= MAX_SECTION:
                        extracted_sections.add(str(i))
                    else:
                        invalid_sections.add(str(i))

        range_matches = re.findall(r'\b(\d+)\s*(?:to|-)\s*(\d+)\b', user_input)
        for start, end in range_matches:
            try:
                start_int = int(start)
                end_int = int(end)
                if start_int > end_int:
                    start_int, end_int = end_int, start_int
                for i in range(start_int, end_int + 1):
                    if MIN_SECTION <= i <= MAX_SECTION:
                        extracted_sections.add(str(i))
                    else:
                        invalid_sections.add(str(i))
            except:
                invalid_sections.add(f"{start}-{end}")

        grouped_matches = re.findall(r'\bsection(?:s)?[^\d]*(\d+(?:[\s,]*(?:and)?[\s,]*\d+)*)', user_input, re.IGNORECASE)
        for group in grouped_matches:
            for num in re.findall(r'\d+', group):
                try:
                    normalized = str(int(num))
                    if MIN_SECTION <= int(normalized) <= MAX_SECTION:
                        extracted_sections.add(normalized)
                    else:
                        invalid_sections.add(normalized)
                except:
                    continue

        all_docs = collection.get()
        all_metadata = all_docs["metadatas"]

        found_exact = []
        formatted_results = []

        for sec_num in extracted_sections:
            matches = [
                meta for meta in all_metadata
                if meta.get("section_number", "").strip() == sec_num
            ]
            found_exact.extend(matches)

        found_exact_sorted = sorted(
            found_exact,
            key=lambda x: int(x.get("section_number", "0").strip())
        )

        for metadata in found_exact_sorted:
            formatted_results.append({
                "law_type": metadata.get('law_type', 'Unknown'),
                "section_summary": metadata.get('section_summary', 'No summary available'),
                "source_url": metadata.get('source_url', 'No source available'),
                "match_type": "Exact"
            })

        if invalid_sections:
            formatted_results.append({
                "law_type": "Invalid",
                "section_summary": f"The following section(s) do not exist: {', '.join(sorted(invalid_sections))} (valid range is 1 to 358).",
                "source_url": "#",
                "match_type": "Error"
            })

        if found_exact:
            context_docs = [meta.get('section_summary', '') for meta in found_exact if meta.get('section_summary')]
            total_context_length = sum(len(doc) for doc in context_docs)

            if total_context_length < 6000:
                groq_answer = generate_response_with_groq(original_query, context_docs, session['chat_history'])

                if not groq_answer.lower().startswith("❌"):
                    formatted_results = []
                    formatted_results.append({
                        "law_type": "Groq AI",
                        "section_summary": groq_answer.strip(),
                        "source_url": "#",
                        "match_type": "Generated"
                    })
                else:
                    print("Groq error skipped due to token limits.")
            else:
                print("Skipping Groq call due to large context size.")
                for metadata in found_exact_sorted:
                    formatted_results.append({
                        "law_type": metadata.get('law_type', 'Unknown'),
                        "section_summary": metadata.get('section_summary', 'No summary available'),
                        "source_url": metadata.get('source_url', 'No source available'),
                        "match_type": "Exact"
                    })

        else:
            # No exact matches: use normalized query for semantic fallback
            expanded_query = rephrase_with_llm(user_input)
            print(f"🔍 Expanded Query: {expanded_query}")
            query_embedding = model.encode(expanded_query).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=15)

            if results and results.get('documents') and results['documents'][0]:
                top_contexts = [doc for doc in results['documents'][0] if doc.strip()]
                
                if top_contexts:
                    groq_answer = generate_response_with_groq(original_query, top_contexts, session['chat_history'])

                    formatted_results.append({
                        "law_type": "Groq AI",
                        "section_summary": groq_answer.strip(),
                        "source_url": "#",
                        "match_type": "Generated"
                    })

                    source_links = []
                    for metadata in results.get('metadatas', [[]])[0]:
                        url = metadata.get('source_url', '')
                        if url and url != '#':
                            source_links.append(url)

                    if source_links:
                        seen = set()
                        unique_links = []
                        for link in source_links:
                            if link not in seen:
                                seen.add(link)
                                unique_links.append(link)

                        formatted_results.append({
                            "law_type": "Sources",
                            "section_summary": "🔗 Sources used for this response:<br>" + "<br>".join(f"- <a href='{link}' target='_blank'>{link}</a>" for link in unique_links),
                            "source_url": "#",
                            "match_type": "Info"
                        })
                else:
                    formatted_results.append({
                        "law_type": "None",
                        "section_summary": "❌ No relevant legal documents were found.",
                        "source_url": "#",
                        "match_type": "Error"
                    })
            else:
                formatted_results.append({
                    "law_type": "None",
                    "section_summary": "❌ No matches found at all.",
                    "source_url": "#",
                    "match_type": "Error"
                })

        result = formatted_results

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session['chat_history'].append({
            "user_input": original_query,
            "response": result,
            "timestamp": timestamp
        })
        session.modified = True

    for chat in session['chat_history']:
        for item in chat["response"]:
            item["section_summary"] = markdown(item["section_summary"])

    return render_template('index.html', result=result, user_input=user_input, chat_history=session['chat_history'])


@app.route('/debug')
def debug():
    all_data = collection.get()
    records = []

    for i in range(len(all_data['ids'])):
        records.append({
            "id": all_data['ids'][i],
            "document": all_data['documents'][i],
            "metadata": all_data['metadatas'][i]
        })

    return render_template('debug.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
