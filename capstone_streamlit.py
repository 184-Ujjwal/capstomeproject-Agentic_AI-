
import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="Student Education Assistant", page_icon="🎓", layout="centered")
st.title("🎓 Student Education & Financing Assistant")
st.caption("Helping Indian students with college choices, scholarships, and education loans.")

@st.cache_resource
def load_agent():
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")

    DOCUMENTS = [
        {"id":"doc_001","topic":"Engineering College Selection Criteria","text":"Choosing the right engineering college in India requires evaluating several key factors. NIRF Ranking published by the Ministry of Education is the most reliable ranking system -- prefer colleges in the top 200 for better placement outcomes. Accreditation by NBA indicates that individual programs meet quality benchmarks, while NAAC accreditation evaluates the overall institution. Student-to-Faculty Ratio below 20:1 is considered good. Government colleges (NITs, IIITs, State Engineering colleges) typically charge Rs 50,000 to Rs 2,00,000 per year in fees; private autonomous colleges range from Rs 1,00,000 to Rs 3,00,000 per year. Lateral entry is available in most states for Diploma holders directly into the second year of B.Tech."},
        {"id":"doc_002","topic":"JEE Main and JEE Advanced Admission Process","text":"JEE Main is the national entrance exam conducted by NTA for admission to NITs, IIITs, and GFTIs. JEE Main is held twice a year -- January session and April session. Eligibility: passed or appearing in Class 12 with Physics, Chemistry, and Mathematics. JEE Advanced is the gateway to IITs -- only top 2,50,000 qualifiers in JEE Main are eligible. IIT admissions happen through JoSAA counselling based on JEE Advanced rank. For state engineering colleges, separate state-level exams like TS EAMCET, AP EAMCET, MHT-CET, KCET, and WBJEE are conducted."},
        {"id":"doc_003","topic":"NEET UG Medical Admission Process","text":"NEET UG is the single national entrance exam for MBBS, BDS, AYUSH, and Nursing admissions across India, conducted by NTA. Eligibility: minimum 50% in Class 12 PCB for general category; 40% for SC/ST/OBC. NEET has 180 questions, total 720 marks. Marking: +4 correct, -1 wrong. Top scorers qualify for 15% All India Quota seats counselled by MCC. Government MBBS seats cost Rs 10,000 to Rs 50,000 per year. Cut-off for government MBBS: approximately 600+ marks for general category AIQ seats."},
        {"id":"doc_004","topic":"Central Sector Scholarship Scheme","text":"The Central Sector Scheme of Scholarships (CSSS) is funded by the Ministry of Education. Eligibility: students who scored above 80th percentile in Class 12 and have annual family income below Rs 8 lakh. The scholarship provides Rs 10,000 per year for first three years of undergraduate study and Rs 20,000 per year for professional courses in 4th and 5th year. Renewal requires minimum 50% marks. Applications through National Scholarship Portal (scholarships.gov.in). The scheme covers approximately 82,000 new scholarships per year."},
        {"id":"doc_005","topic":"Education Loan from Public Sector Banks","text":"Education loans from public sector banks follow IBA model education loan scheme guidelines. Loans up to Rs 4 lakh require no collateral. Loans above Rs 7.5 lakh require tangible collateral security. Interest rates range from 8.5% to 11.5% per annum. Repayment starts after course completion plus moratorium of 6 to 12 months. Maximum repayment tenure is 15 years. Section 80E of Income Tax Act allows full deduction on interest paid for up to 8 years. Vidya Lakshmi portal (vidyalakshmi.co.in) allows applying to multiple banks through a single form."},
        {"id":"doc_006","topic":"PM Vidyalaxmi Scheme","text":"PM Vidyalaxmi provides collateral-free, guarantor-free education loans to meritorious students admitted to top quality higher education institutions ranked in NIRF top 100 or having NAAC A+ grade. Loans up to Rs 10 lakh are covered with government credit guarantee. For students with annual family income up to Rs 8 lakh, a 3% interest subvention is provided. For family income between Rs 8 lakh and Rs 15 lakh, 1% interest subvention is available. Students apply through the PM Vidyalaxmi portal linked with Aadhaar."},
        {"id":"doc_007","topic":"MBA Entrance Exams and Admission Process","text":"MBA admissions in India are primarily through CAT for IIMs. CAT is conducted in November/December. It tests VARC, DILR, and QA. Beyond CAT: XAT for XLRI; SNAP for Symbiosis; MAT accepted by 600+ B-schools; CMAT for AICTE-approved institutes. IIM fees range from Rs 20 lakh to Rs 27 lakh for the full 2-year program. Average salary from top IIMs ranges from Rs 20 lakh to Rs 35 lakh per annum. Eligibility for CAT: any bachelor's degree with minimum 50% aggregate."},
        {"id":"doc_008","topic":"State Scholarships for Telangana and Andhra Pradesh","text":"Telangana government offers TS ePass scholarship for students from SC, ST, BC, EBC, and Minority communities. SC students with family income below Rs 2.5 lakh get full fee reimbursement plus maintenance allowance. For Andhra Pradesh, the AP ePass portal manages scholarships for SC/ST/BC/EBC/Minority/Disabled students. The Post Matric Scholarship (PMS) from central government applies to SC/ST students for any post-Class 10 education. Students must submit Aadhaar, income certificate, caste certificate, admission letter, and bank passbook."},
        {"id":"doc_009","topic":"Vocational Courses and Diploma After Class 10","text":"Government ITIs offer 1-year and 2-year trade certificates in Electrician, Fitter, Welder, Mechanic, COPA, and Draughtsman trades. Polytechnic Diploma (3 years after Class 10) leads to Diploma in Engineering. Diploma holders can enter B.Tech directly in second year through lateral entry. PMKVY short-term courses (3-6 months) are free and offer stipends. Average salary after ITI: Rs 12,000 to Rs 25,000 per month."},
        {"id":"doc_010","topic":"Study Abroad Funding and Scholarships","text":"Education loans for foreign studies from banks go up to Rs 1.5 crore. NBFCs like Prodigy Finance offer loans without collateral. Scholarships: DAAD (Germany) offers fully-funded scholarships; Chevening (UK) is fully-funded; Fulbright-Nehru (USA) for research students. National Overseas Scholarship (NOS) provides up to USD 15,400 per year for SC/ST students. IELTS 6.5+ and TOEFL 90+ are required for UK and US admissions. Germany offers tuition-free public universities."},
        {"id":"doc_011","topic":"TS EAMCET Admission Process","text":"TS EAMCET is conducted by JNTU Hyderabad for B.Tech and B.Pharmacy in Telangana. Eligibility: minimum 40% marks in Class 12 with MPC for general category; no minimum for SC/ST. Exam pattern: 160 questions -- 80 Mathematics, 40 Physics, 40 Chemistry -- no negative marking. Rank is calculated as 75% EAMCET score and 25% Class 12 marks. Government engineering college fees in Telangana: Rs 35,000 to Rs 55,000 per year."},
        {"id":"doc_012","topic":"Education Loan Repayment and Interest Subvention","text":"Education loan repayment begins after moratorium -- course duration plus 6-12 months. The Dr. Ambedkar Interest Subsidy Scheme provides full interest subsidy for OBC/EBC students studying abroad with family income below Rs 3 lakh. The Central Government Interest Subsidy Scheme (CGISS) gives full interest subsidy during moratorium for students with family income below Rs 4.5 lakh. Students should apply for subsidy schemes through their bank at time of loan disbursement."},
    ]
    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    class CapstoneState(TypedDict):
        question:     str
        messages:     List[dict]
        route:        str
        retrieved:    str
        sources:      List[str]
        tool_result:  str
        answer:       str
        faithfulness: float
        eval_retries: int
        user_name:    str

    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2

    def memory_node(state):
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6: msgs = msgs[-6:]
        user_name = state.get("user_name", "")
        if "my name is" in state["question"].lower():
            try: user_name = state["question"].lower().split("my name is")[1].strip().split()[0].capitalize()
            except: pass
        return {"messages": msgs, "user_name": user_name}

    def router_node(state):
        q = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"
        prompt = f"Router for Student Education assistant. Options: retrieve (knowledge base), memory_only (history), tool (web search for live info). Recent: {recent}. Question: {q}. Reply ONLY: retrieve / memory_only / tool"
        decision = llm.invoke(prompt).content.strip().lower()
        if "memory" in decision: decision = "memory_only"
        elif "tool" in decision: decision = "tool"
        else: decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "

---

".join(f"[{topics[i]}]
{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        question = state["question"]
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(question + " India education 2025 2026", max_results=3))
            tool_result = "
".join(f"* {r['title']}: {r['body'][:200]}" for r in results) if results else f"No results for: {question}."
        except Exception as e:
            tool_result = f"Web search unavailable ({e}). Check scholarships.gov.in."
        return {"tool_result": tool_result}

    def answer_node(state):
        question = state["question"]; retrieved = state.get("retrieved",""); tool_result = state.get("tool_result","")
        messages = state.get("messages",[]); eval_retries = state.get("eval_retries",0); user_name = state.get("user_name","")
        context_parts = []
        if retrieved: context_parts.append(f"KNOWLEDGE BASE:
{retrieved}")
        if tool_result: context_parts.append(f"WEB SEARCH:
{tool_result}")
        context = "

".join(context_parts)
        name_line = f" Address the student as {user_name}." if user_name else ""
        if context:
            sys = f"You are a Student Education and Financing Assistant for India.{name_line} Answer using ONLY the context below. If not in context, say: I do not have that information. Please check the official portal.

{context}"
        else:
            sys = f"You are a Student Education and Financing Assistant for India.{name_line} Answer from conversation history."
        if eval_retries > 0: sys += "

Use ONLY the context. Do not add anything beyond it."
        lc_msgs = [SystemMessage(content=sys)]
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"]=="user" else AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))
        return {"answer": llm.invoke(lc_msgs).content}

    def eval_node(state):
        answer = state.get("answer",""); context = state.get("retrieved","")[:500]; retries = state.get("eval_retries",0)
        if not context: return {"faithfulness": 1.0, "eval_retries": retries+1}
        try:
            score = float(llm.invoke(f"Rate faithfulness 0.0-1.0 only a number.
Context: {context}
Answer: {answer[:300]}").content.strip().split()[0].replace(",","."))
            score = max(0.0, min(1.0, score))
        except: score = 0.5
        return {"faithfulness": score, "eval_retries": retries+1}

    def save_node(state):
        return {"messages": state.get("messages",[]) + [{"role":"assistant","content":state["answer"]}]}

    def route_decision(state):
        r = state.get("route","retrieve")
        if r=="tool": return "tool"
        if r=="memory_only": return "skip"
        return "retrieve"

    def eval_decision(state):
        if state.get("faithfulness",1.0) >= FAITHFULNESS_THRESHOLD or state.get("eval_retries",0) >= MAX_EVAL_RETRIES: return "save"
        return "answer"

    g = StateGraph(CapstoneState)
    for name, fn in [("memory",memory_node),("router",router_node),("retrieve",retrieval_node),
                     ("skip",skip_retrieval_node),("tool",tool_node),("answer",answer_node),
                     ("eval",eval_node),("save",save_node)]:
        g.add_node(name, fn)
    g.set_entry_point("memory")
    g.add_edge("memory","router")
    g.add_conditional_edges("router", route_decision, {"retrieve":"retrieve","skip":"skip","tool":"tool"})
    g.add_edge("retrieve","answer"); g.add_edge("skip","answer"); g.add_edge("tool","answer")
    g.add_edge("answer","eval")
    g.add_conditional_edges("eval", eval_decision, {"answer":"answer","save":"save"})
    g.add_edge("save", END)
    return g.compile(checkpointer=MemorySaver()), embedder, collection


try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded -- {collection.count()} documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}"); st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About")
    st.write("Helping Indian students with college choices, scholarships, and education loans.")
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    for t in ["Engineering & Medical Admissions","JEE Main / JEE Advanced / NEET","TS EAMCET","Central & State Scholarships","Education Loans & PM Vidyalaxmi","MBA Entrance Exams","Vocational & Diploma Courses","Study Abroad Funding","Loan Repayment & Subsidies"]:
        st.write(f"- {t}")
    if st.button("New conversation"):
        st.session_state.messages = []; st.session_state.thread_id = str(uuid.uuid4())[:8]; st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ask about colleges, scholarships, loans, entrance exams..."):
    with st.chat_message("user"): st.write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0: st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}")
    st.session_state.messages.append({"role":"assistant","content":answer})
