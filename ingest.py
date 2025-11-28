import os
import re
import chromadb
from chromadb.utils import embedding_functions
import pymupdf  # PyMuPDF (more reliable than pymdownx)
from typing import List, Dict, Tuple

# Configuration
CHUNK_SIZE = 500  # Smaller chunks for better retrieval
CHUNK_OVERLAP = 100  # More overlap to preserve context
MIN_CHUNK_LENGTH = 100  # Skip very short chunks

vector_client = chromadb.PersistentClient(path="./tata_knowledge_base")

sentence_transformer_embedding = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

collection = vector_client.get_or_create_collection(
    name="financial_knowledge_base", embedding_function=sentence_transformer_embedding
)


def clean_text(text: str) -> str:
    """Clean extracted text while preserving structure."""
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r"\n\d+\n", "\n", text)
    return text.strip()


def table_to_markdown(table_data: List[List[str]]) -> str:
    """Convert table data to properly formatted markdown."""
    if not table_data or len(table_data) < 2:
        return ""

    # Clean cells
    clean_data = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in table_data
    ]

    # Calculate column widths for alignment
    col_widths = [
        max(len(str(row[i])) for row in clean_data) for i in range(len(clean_data[0]))
    ]

    # Header row
    markdown = "| " + " | ".join(clean_data[0]) + " |\n"
    # Separator row
    markdown += "| " + " | ".join(["---" for _ in clean_data[0]]) + " |\n"

    # Data rows
    for row in clean_data[1:]:
        markdown += "| " + " | ".join(row) + " |\n"

    return markdown


def smart_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Intelligently chunk text by respecting sentence and paragraph boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            overlap_text = (
                current_chunk[-overlap:]
                if len(current_chunk) > overlap
                else current_chunk
            )
            current_chunk = overlap_text + "\n\n" + para
        else:
            current_chunk += ("\n\n" if current_chunk else "") + para

    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def extract_tables_from_page(page) -> List[Tuple[str, Dict]]:
    """Extract all tables from a page with metadata."""
    tables_data = []

    try:
        tables = page.find_tables()

        for i, table in enumerate(tables):
            # Skip external headers
            if (
                hasattr(table, "header")
                and hasattr(table.header, "external")
                and table.header.external
            ):
                continue

            table_data = table.extract()
            if table_data and len(table_data) > 1:  # Must have header + at least 1 row
                table_text = table_to_markdown(table_data)

                if table_text:
                    # Add context header
                    full_chunk = f"**Table {i+1}**\n\n{table_text}"

                    metadata = {
                        "type": "table",
                        "table_index": i,
                        "row_count": len(table_data),
                        "col_count": len(table_data[0]) if table_data else 0,
                    }

                    tables_data.append((full_chunk, metadata))

    except Exception as e:
        print(f"Warning: Error extracting tables: {e}")

    return tables_data


def process_pdf(file_path: str) -> Dict[str, int]:
    """
    Process PDF with improved parsing and chunking.
    Returns statistics about the ingestion.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    print(f"Processing: {file_path}")

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

    documents = []
    metadatas = []
    ids = []

    stats = {"pages": 0, "tables": 0, "text_chunks": 0, "total_chunks": 0}

    for page_num in range(len(doc)):
        page = doc[page_num]
        stats["pages"] += 1

        # Extract tables
        tables_data = extract_tables_from_page(page)

        for i, (table_text, table_meta) in enumerate(tables_data):
            documents.append(table_text)
            metadatas.append(
                {
                    "source": file_path,
                    "page": page_num + 1,
                    "type": "table",
                    **table_meta,
                }
            )
            ids.append(f"{os.path.basename(file_path)}_p{page_num+1}_table{i+1}")
            stats["tables"] += 1

        # Extract and clean text
        try:
            raw_text = page.get_text()
            clean_text_content = clean_text(raw_text)

            if len(clean_text_content) > MIN_CHUNK_LENGTH:
                # Smart chunking
                text_chunks = smart_chunk_text(
                    clean_text_content, CHUNK_SIZE, CHUNK_OVERLAP
                )

                for j, chunk in enumerate(text_chunks):
                    if len(chunk) >= MIN_CHUNK_LENGTH:
                        documents.append(chunk)
                        metadatas.append(
                            {
                                "source": file_path,
                                "page": page_num + 1,
                                "type": "text",
                                "chunk_index": j,
                                "total_chunks_on_page": len(text_chunks),
                            }
                        )
                        ids.append(
                            f"{os.path.basename(file_path)}_p{page_num+1}_chunk{j+1}"
                        )
                        stats["text_chunks"] += 1

        except Exception as e:
            print(f"Warning: Error processing page {page_num + 1}: {e}")
            continue

    doc.close()

    stats["total_chunks"] = len(documents)

    print(f"\nExtraction Summary:")
    print(f"  - Pages processed: {stats['pages']}")
    print(f"  - Tables extracted: {stats['tables']}")
    print(f"  - Text chunks created: {stats['text_chunks']}")
    print(f"  - Total chunks: {stats['total_chunks']}")

    # Ingest into ChromaDB in batches
    print("\nIngesting into Vector DB...")

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        try:
            collection.upsert(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            print(f"  [OK] Batch {i//batch_size + 1}: {batch_end - i} chunks")
        except Exception as e:
            print(f"  [ERROR] Batch {i//batch_size + 1}: {e}")

    print(f"\nIngestion complete!")
    return stats


def test_retrieval(query: str, n_results: int = 3):
    """Test the retrieval quality."""
    print(f"\nTesting retrieval for: '{query}'")
    print("=" * 60)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        print(f"\nResult {i+1} [{metadata['type'].upper()}] - Page {metadata['page']}")
        print(f"   Source: {metadata.get('source', 'N/A')}")

        # Show preview
        preview = doc[:400] if len(doc) > 400 else doc
        print(f"\n{preview}")
        if len(doc) > 400:
            print("\n   [...truncated...]")
        print("-" * 60)


if __name__ == "__main__":
    # Process the PDF
    pdf_file_name = "tata-motor-IAR-2024-25.pdf"

    try:
        stats = process_pdf(pdf_file_name)

        # Run test queries
        print("\n" + "=" * 60)
        print("TESTING RETRIEVAL QUALITY")
        print("=" * 60)

        test_queries = [
            "Consolidated Balance Sheet Assets",
            "revenue growth",
            "total liabilities",
        ]

        for query in test_queries:
            test_retrieval(query, n_results=2)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
