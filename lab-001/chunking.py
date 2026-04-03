import pymupdf
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunking():
    print("This is a placeholder for the chunking function.")
    pdf_chunking()


def pdf_chunking():
    print("This is a placeholder for the PDF chunking function.")
    # sentence_based_chunking(file="data/small.pdf") - can read
    # sentence_based_chunking(file="data/medium.pdf") - can read
    # sentence_based_chunking(file="data/large.pdf") - can not read
    # page_based_chunking_with_pymupdf(file="data/small.pdf")  # can read
    # page_based_chunking_with_pymupdf(file="data/medium.pdf")  # can read
    # page_based_chunking_with_pymupdf(file="data/large.pdf")  # can read but first page is empty
    # recursive_chunking(file="data/small.pdf")  # can read
    # recursive_chunking(file="data/medium.pdf")  # can read
    # recursive_chunking(file="data/large.pdf")  # can read

    # why it is good
    # retrieved_chunks = [chunk_5, chunk_6]
    # fetch chunk_4, chunk_7 (neighbors) -> context window expansion
    # orders Chunk 5 → Chunk 6 → Chunk 7


def recursive_chunking(file="data/small.pdf"):
    doc = pymupdf.open(file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    records = []
    global_chunk_index = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True).strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for page_chunk_index, chunk in enumerate(chunks):
            records.append({
                "text": chunk,
                "metadata": {
                    "source": file,
                    "page": page_num,
                    "chunk_index": global_chunk_index,
                    "page_chunk_index": page_chunk_index,
                }
            })
            global_chunk_index += 1

    print(records)


def page_based_chunking_with_pymupdf(file="data/small.pdf"):
    doc = pymupdf.open(file)  # open a document
    print(f"Number of pages: {doc.page_count}")
    print(f"Toc: {doc.get_toc()}")
    # print the first 100 characters of the first page
    print(f"First page text: {doc[0].get_text()}")
    # print the first 100 characters of the middle page
    print(f"Middle page text: {doc[doc.page_count // 2].get_text()}")
    print(f"Last page text: {doc[-1].get_text()}")


def sentence_based_chunking(file="data/small.pdf"):
    print("This is a placeholder for the small PDF chunking function.")
    reader = PdfReader(file)
    number_of_pages = len(reader.pages)
    print(f"Number of pages: {number_of_pages}")
    page = reader.pages[0]
    text = page.extract_text()
    # print(f"Text from the first page: {text}")
    # print(f"Length of the text: {len(text)} characters")
    # print(f"Number of words: {len(text.split())}")
    # print(f"Number of sentences: {len(text.split('.'))}")
    # meta = reader.metadata

    # # All the following could be None!
    # print(f"Title: {meta.title}")
    # print(f"Author: {meta.author}")
    # print(f"Subject: {meta.subject}")
    # print(f"Creator: {meta.creator}")
    # print(f"Producer: {meta.producer}")
    # print(f"Creation Date: {meta.creation_date}")
    # print(f"Modification Date: {meta.modification_date}")
    # let's chunk by sentences
    sentences = text.split(".")
    print(f"Number of sentences: {len(sentences)}")
    print(f"First sentence: {sentences[0]}")
    print(f"Middle sentence: {sentences[len(sentences) // 2]}")
    print(f"Last sentence: {sentences[-1]}")


if __name__ == "__main__":
    chunking()
