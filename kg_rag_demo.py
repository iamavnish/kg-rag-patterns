"""
Knowledge Graph RAG — Learning Demo
====================================
Demonstrates 3 RAG patterns side-by-side on a NetApp SEC 10-K filing:
  Pattern 1 — Plain Vector RAG
  Pattern 2 — Window Retrieval (expands context via NEXT relationships)
  Pattern 4 — LLM-Generated Cypher (GraphCypherQAChain)

LLM    : Claude Haiku (Anthropic) — requires ANTHROPIC_API_KEY
Embed  : all-MiniLM-L6-v2 (local, free, 384-dim, runs on CPU)
Graph  : Neo4j AuraDB Free Tier — requires NEO4J_* env vars
"""

import json
import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain, GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

FORM_10K_PATH = Path(__file__).parent / "0000950170-23-027948.json"
FORM_ID       = "0000950170-23-027948"
SECTIONS      = ["item1"]   # only item1 has substantial content in this filing
MAX_CHUNKS    = 10          # 10 chunks × ~2000 chars from the business section
VECTOR_DIM    = 384         # all-MiniLM-L6-v2 output dimension
INDEX_NAME    = "form_10k_chunks"

# Claude Haiku — cheapest Anthropic model, ~$0.001 per question asked
LLM = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# Local embedding model — free, 22 MB, runs on CPU, no API key needed
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ── Step 1: Graph Setup (idempotent — safe to re-run) ─────────────────────────

def setup_graph(kg: Neo4jGraph) -> None:
    print("  [1/4] Creating schema constraints and vector index...")

    kg.query("""
        CREATE CONSTRAINT unique_chunk IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
    """)

    kg.query(f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON (c.textEmbedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
    """)

    print("  [2/4] Loading and chunking 10-K text...")
    with open(FORM_10K_PATH, encoding="utf-8") as f:
        form_data = json.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # Form node (the 10-K filing itself)
    kg.query("""
        MERGE (f:Form {formId: $formId})
        ON CREATE SET f.source = $source, f.names = $names
    """, params={
        "formId": FORM_ID,
        "source": form_data.get("source", ""),
        "names":  form_data.get("names", []),
    })

    for section in SECTIONS:
        text = form_data.get(section, "")
        if not text:
            continue

        chunks = splitter.split_text(text)[:MAX_CHUNKS]

        for seq_id, chunk_text in enumerate(chunks):
            chunk_id = f"{FORM_ID}-{section}-chunk{seq_id:04d}"
            kg.query("""
                MERGE (c:Chunk {chunkId: $chunkId})
                ON CREATE SET
                    c.text       = $text,
                    c.f10kItem   = $f10kItem,
                    c.chunkSeqId = $chunkSeqId,
                    c.formId     = $formId,
                    c.source     = $source
            """, params={
                "chunkId":    chunk_id,
                "text":       chunk_text,
                "f10kItem":   section,
                "chunkSeqId": seq_id,
                "formId":     FORM_ID,
                "source":     form_data.get("source", ""),
            })

        # NEXT relationships — ordered chain within this section (pure Cypher, no APOC)
        kg.query("""
            MATCH (c1:Chunk {f10kItem: $section, formId: $formId}),
                  (c2:Chunk {f10kItem: $section, formId: $formId})
            WHERE c2.chunkSeqId = c1.chunkSeqId + 1
            MERGE (c1)-[:NEXT]->(c2)
        """, params={"section": section, "formId": FORM_ID})

    # PART_OF — each Chunk belongs to its Form
    kg.query("""
        MATCH (c:Chunk {formId: $formId}), (f:Form {formId: $formId})
        MERGE (c)-[:PART_OF]->(f)
    """, params={"formId": FORM_ID})

    # SECTION — Form points to the first Chunk of each section
    kg.query("""
        MATCH (c:Chunk {formId: $formId, chunkSeqId: 0}),
              (f:Form  {formId: $formId})
        MERGE (f)-[r:SECTION {f10kItem: c.f10kItem}]->(c)
    """, params={"formId": FORM_ID})

    print("  ✓ Graph nodes and relationships ready.")


# ── Step 2: Embed Chunks (idempotent — skips chunks already embedded) ─────────

def populate_embeddings(kg: Neo4jGraph) -> None:
    print("  [3/4] Checking embeddings...")
    pending = kg.query("""
        MATCH (c:Chunk) WHERE c.textEmbedding IS NULL
        RETURN c.chunkId AS id, c.text AS text
    """)

    if not pending:
        print("  ✓ Embeddings already populated — skipping.")
        return

    print(f"  Generating {len(pending)} embeddings (local model, no API cost)...")
    for chunk in pending:
        vector = EMBEDDINGS.embed_query(chunk["text"])
        kg.query("""
            MATCH (c:Chunk {chunkId: $id})
            SET c.textEmbedding = $vec
        """, params={"id": chunk["id"], "vec": vector})

    print(f"  ✓ Embedded {len(pending)} chunks.")


# ── Step 3: Build All Three RAG Chains ────────────────────────────────────────

def build_chains(kg: Neo4jGraph):
    print("  [4/4] Building RAG chains...")

    neo4j_params = dict(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    # ── Pattern 1: Plain Vector RAG ───────────────────────────────────────────
    # Finds the single most similar chunk by cosine similarity, feeds it to LLM.
    store1 = Neo4jVector.from_existing_index(
        embedding=EMBEDDINGS,
        index_name=INDEX_NAME,
        text_node_property="text",
        embedding_node_property="textEmbedding",
        **neo4j_params,
    )
    chain1 = RetrievalQAWithSourcesChain.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=store1.as_retriever(),
    )

    # ── Pattern 2: Window Retrieval RAG ──────────────────────────────────────
    # Same vector search, but then expands to neighbouring chunks (NEXT links).
    # Uses reduce() for string join — pure Cypher, no APOC plugin needed.
    window_query = """
        MATCH window=(:Chunk)-[:NEXT*0..1]->(node)-[:NEXT*0..1]->(:Chunk)
        WITH node, score, window AS longestWindow
          ORDER BY length(window) DESC LIMIT 1
        WITH nodes(longestWindow) AS chunkList, node, score
        UNWIND chunkList AS row
        WITH collect(row.text) AS textList, node, score
        RETURN reduce(s = '', t IN textList | s + t + ' \\n ') AS text,
               score,
               node {.source} AS metadata
    """
    store2 = Neo4jVector.from_existing_index(
        embedding=EMBEDDINGS,
        index_name=INDEX_NAME,
        text_node_property="text",
        embedding_node_property="textEmbedding",
        retrieval_query=window_query,
        **neo4j_params,
    )
    chain2 = RetrievalQAWithSourcesChain.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=store2.as_retriever(),
    )

    # ── Pattern 4: LLM-Generated Cypher ──────────────────────────────────────
    # LLM receives the graph schema + few-shot examples → generates Cypher →
    # Neo4j executes it → LLM converts raw results into a natural-language answer.
    cypher_template = """Generate a Cypher statement to query a Neo4j graph database.
Instructions:
- Use only the node labels, relationship types, and properties in the schema below.
- Do not include any explanations or apologies.
- Return ONLY the Cypher statement, nothing else.

Schema:
{schema}

Examples of correct Cypher for this graph:

# What does NetApp do?
MATCH (f:Form)-[s:SECTION]->(c:Chunk)
WHERE s.f10kItem = 'item1'
RETURN c.text LIMIT 3

# What is in the very first chunk of the document?
MATCH (c:Chunk {{chunkSeqId: 0}})
RETURN c.text

# How many text chunks does this filing have?
MATCH (c:Chunk)
RETURN count(c) AS totalChunks

The question is:
{question}"""

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=cypher_template,
    )
    chain4 = GraphCypherQAChain.from_llm(
        llm=LLM,
        graph=kg,
        verbose=True,
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True,
    )

    print("  ✓ All chains ready.\n")
    return chain1, chain2, chain4


# ── Main: Interactive Question Loop ───────────────────────────────────────────

def ask(chain1, chain2, chain4, question: str) -> None:
    DIVIDER = "═" * 60

    print(f"\n{DIVIDER}")
    print("  PATTERN 1 — Plain Vector RAG")
    print("  (Finds the single most similar chunk by embedding)")
    print(DIVIDER)
    try:
        r1 = chain1({"question": question}, return_only_outputs=True)
        print(textwrap.fill(r1.get("answer", "No answer.").strip(), 70))
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{DIVIDER}")
    print("  PATTERN 2 — Window Retrieval RAG")
    print("  (Expands to include neighbouring chunks via NEXT links)")
    print(DIVIDER)
    try:
        r2 = chain2({"question": question}, return_only_outputs=True)
        print(textwrap.fill(r2.get("answer", "No answer.").strip(), 70))
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{DIVIDER}")
    print("  PATTERN 4 — LLM-Generated Cypher")
    print("  (LLM writes a Cypher query, runs it, then answers)")
    print(DIVIDER)
    try:
        chain4.run(question)
    except Exception as e:
        print(f"  Error: {e}")

    print()


def main() -> None:
    print("\n" + "─" * 60)
    print("  Knowledge Graph RAG — Learning Demo")
    print("─" * 60)

    kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

    print("\nInitialising (idempotent — safe to re-run)...")
    setup_graph(kg)
    populate_embeddings(kg)
    chain1, chain2, chain4 = build_chains(kg)

    print("✅  Ready! Each question runs through all 3 RAG patterns.")
    print("    Suggested test questions:")
    print("    Test 1 → \"Where is NetApp headquartered?\"")
    print("    Test 2 → \"Give me a detailed overview of NetApp's business and cloud strategy\"")
    print("    Test 3 → \"How many text chunks does this filing have?\"")
    print("    Type 'quit' to exit.\n")

    while True:
        try:
            question = input("💬  Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break

        ask(chain1, chain2, chain4, question)


if __name__ == "__main__":
    main()
