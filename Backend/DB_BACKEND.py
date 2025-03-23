"""
DSA Educational RAG System
- Processes PDFs to generate Q&A pairs using Claude API
- Stores Q&A pairs in a persistent Chroma vector database
- Provides RAG-based student guidance using DeepSeek
"""

import os
import json
import time
import re
import argparse
import logging
from typing import List, Dict, Any, Tuple

# Required libraries
import anthropic
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dsa_rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MIN_CHUNK_SIZE = 1000  # Minimum characters in a chunk
MAX_CHUNK_SIZE = 3000  # Maximum characters in a chunk
CHUNK_OVERLAP = 200    # Overlap between chunks to maintain context
QUESTIONS_PER_PAGE = 5  # Target number of Q&A pairs per page
CHROMA_PERSIST_DIR = "chroma_db"  # Directory for persistent Chroma DB

# Claude API setup
def setup_claude_client(api_key: str) -> anthropic.Anthropic:
    """Initialize and return Claude API client."""
    return anthropic.Anthropic(api_key=api_key)

# PDF text extraction and page counting
def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extract text from a PDF file and return the text and page count."""
    logger.info(f"Extracting text from {pdf_path}")
    
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        page_count = len(reader.pages)
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        logger.info(f"Successfully extracted {len(text)} characters from {page_count} pages")
        return text, page_count
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

# Text chunking
def split_text_into_chunks(text: str) -> List[str]:
    """Split text into manageable chunks for API processing."""
    logger.info("Splitting text into chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    
    # Filter out chunks that are too small
    filtered_chunks = [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_SIZE]
    logger.info(f"After filtering, {len(filtered_chunks)} chunks remain")
    
    return filtered_chunks

# Extract topic from chunk
def extract_topic(chunk: str) -> str:
    """Try to extract the main topic from a chunk."""
    # Look for section headings or chapter titles
    lines = chunk.split('\n')
    for line in lines[:5]:  # Check first few lines
        line = line.strip()
        # If line is short and uppercase or ends with a colon, it might be a heading
        if (len(line) < 60 and (line.isupper() or line.endswith(':') or 
            re.match(r'^(Chapter|Section|\d+\.|\d+\.\d+)', line))):
            return line
    
    # If no clear heading, use first non-empty line
    for line in lines:
        if line.strip():
            # Truncate if too long
            return line.strip()[:50] + ('...' if len(line) > 50 else '')
    
    # Fallback
    return "CS Topic"

# Generate questions from a chunk using Claude
def generate_questions_from_chunk(
    client: anthropic.Anthropic, 
    chunk: str, 
    topic: str, 
    num_questions: int = 3
) -> List[str]:
    """Generate potential questions from a chunk of text using Claude."""
    logger.info(f"Generating {num_questions} questions for topic: {topic}")
    
    prompt = f"""Based on the following text from a Computer Science textbook, 
generate {num_questions} clear, specific questions that a student might ask. 
The questions should be directly related to the concepts in the text and should be
the kind that would typically expect a detailed educational response.

Text from textbook:
```
{chunk}
```

Generate exactly {num_questions} questions. Format them as a numbered list without 
explanations or additional text. Each question should be specific enough for an educational context.
"""
    
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system="You are an expert in computer science education, specializing in creating educational content.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        # Extract questions using regex
        questions = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', response_text, re.DOTALL)
        
        # Clean up questions
        questions = [q.strip() for q in questions]
        
        # If regex didn't work well, just split by newlines and try to clean up
        if not questions or len(questions) < num_questions:
            questions = [line.strip() for line in response_text.split('\n') 
                         if line.strip() and not line.strip().startswith('```')]
            questions = [re.sub(r'^\d+\.\s*', '', q) for q in questions]
        
        logger.info(f"Generated {len(questions)} questions")
        return questions[:num_questions]  # Return only the requested number
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return [f"Explain the concept of {topic}"]  # Fallback question

# Generate roadmap answer for a question
def generate_roadmap_answer(
    client: anthropic.Anthropic,
    question: str,
    chunk: str
) -> str:
    """Generate a roadmap answer instead of a direct answer."""
    logger.info(f"Generating roadmap answer for question: {question}")
    
    prompt = f"""Instead of directly answering this computer science question,
create a learning roadmap that guides the student to discover the answer themselves.

Question: {question}

Context from textbook:
```
{chunk}
```

Your response should:
1. Start with "To understand [topic], follow this learning path: ..."
2. Provide a step-by-step learning path with 5-7 clear steps
3. Recommend specific resources and materials
4. Include learning activities or exercises
5. Give a hint that points toward the answer without fully revealing it

DO NOT provide the direct answer. Focus on creating a roadmap for learning.
"""
    
    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",  # Using Sonnet for higher quality responses
            max_tokens=1500,
            temperature=0.7,
            system="You are an expert computer science educator who creates learning roadmaps rather than giving direct answers.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        logger.info(f"Generated roadmap answer of length {len(response_text)}")
        return response_text
        
    except Exception as e:
        logger.error(f"Error generating roadmap answer: {e}")
        return "I recommend studying this topic step-by-step using your textbook."  # Fallback answer

# Setup Chroma vector store
def setup_chroma_db() -> chromadb.PersistentClient:
    """Set up and return a persistent Chroma DB client."""
    logger.info(f"Setting up persistent Chroma DB at {CHROMA_PERSIST_DIR}")
    
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    # Initialize client with sentence transformer for embeddings
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Create or get the collection
    # Using SentenceTransformer embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = client.get_or_create_collection(
            name="cs_educational_qa",
            embedding_function=sentence_transformer_ef,
            metadata={"description": "Computer Science educational Q&A pairs"}
        )
        logger.info(f"Collection ready with {collection.count()} existing documents")
        return client
    except Exception as e:
        logger.error(f"Error setting up Chroma DB: {e}")
        raise

# Add Q&A pairs to Chroma
def add_qa_to_chroma(
    chroma_client: chromadb.PersistentClient,
    qa_pairs: List[Dict[str, str]],
    pdf_name: str
) -> None:
    """Add Q&A pairs to the Chroma vector store in smaller batches."""
    logger.info(f"Adding {len(qa_pairs)} Q&A pairs to Chroma DB")
    
    collection = chroma_client.get_collection("cs_educational_qa")
    
    # Maximum batch size for Chroma
    MAX_BATCH_SIZE = 150
    
    # Process in batches
    total_added = 0
    for i in range(0, len(qa_pairs), MAX_BATCH_SIZE):
        batch = qa_pairs[i:i+MAX_BATCH_SIZE]
        
        # Prepare data for batch insertion
        ids = []
        questions = []
        answers = []
        metadatas = []
        
        for j, qa_pair in enumerate(batch):
            # Create a unique ID
            unique_id = f"{pdf_name}_{int(time.time())}_{i+j}"
            
            ids.append(unique_id)
            questions.append(qa_pair["input"])
            # Store the full Q&A pair as document
            full_document = f"Question: {qa_pair['input']}\n\nRoadmap: {qa_pair['output']}"
            answers.append(full_document)
            metadatas.append({
                "source": pdf_name,
                "type": "educational_roadmap",
                "timestamp": time.time()
            })
        
        # Add batch to collection
        try:
            collection.add(
                ids=ids,
                documents=answers,
                metadatas=metadatas,
                embeddings=None  # Let the embedding function handle this
            )
            total_added += len(batch)
            logger.info(f"Added batch of {len(batch)} Q&A pairs ({total_added}/{len(qa_pairs)} total)")
        except Exception as e:
            logger.error(f"Error adding batch to Chroma DB: {e}")
            raise
    
    logger.info(f"Successfully added all {total_added} Q&A pairs to Chroma DB")

# Set up DeepSeek model for GPU-accelerated RAG inference
def setup_deepseek_model():
    """Set up the DeepSeek model for GPU-accelerated RAG inference."""
    logger.info("Loading DeepSeek model for RAG inference with GPU optimization")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            
            # Configure quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Load model with GPU optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",  # Automatically map to available GPUs
                torch_dtype=torch.float16,  # Use half precision
                trust_remote_code=True
            )
            
            # Enable GPU optimizations
            if hasattr(model, 'enable_llm_optimizations'):
                model.enable_llm_optimizations()
            
            logger.info("DeepSeek model loaded successfully with GPU optimizations")
        else:
            logger.warning("No GPU detected, falling back to CPU (slower inference)")
            
            # CPU fallback
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
            
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading DeepSeek model: {e}")
        raise

# Query RAG system with GPU optimization
def query_rag_system(
    query: str,
    chroma_client: chromadb.PersistentClient,
    model,
    tokenizer,
    top_k: int = 5  # Increased from 3 to 5 for better context
) -> str:
    """Query the RAG system with a student question using GPU acceleration."""
    logger.info(f"Querying RAG system: {query}")
    
    # Get the collection
    collection = chroma_client.get_collection("cs_educational_qa")
    
    # Query for relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Extract the retrieved documents
    retrieved_docs = results["documents"][0]
    
    # Combine retrieved contexts
    context = "\n\n".join(retrieved_docs)
    
    # Build prompt for DeepSeek
    system_prompt = """You are an expert in computer science education. Your role is to help students 
learn by providing guidance, not direct answers. Respond with learning roadmaps, hints, and resources 
that guide the student to discover the answer themselves. Never solve problems directly for students."""
    
    example_input = "How does the merge sort algorithm work?"
    
    example_output = """To understand the merge sort algorithm, follow this learning path:

1. Review the divide-and-conquer paradigm in algorithm design
   - Read the "Algorithm Design Paradigms" chapter in your textbook
   - Watch MIT's OCW lecture on divide-and-conquer strategies

2. Study the conceptual foundation of merge sort
   - Visualize how splitting and merging works with simple arrays
   - Draw the recursion tree for a small example (e.g., [5,2,4,7,1,3])

3. Analyze the merge operation in detail
   - Implement the merge function that combines two sorted arrays
   - Test it with various input cases

4. Implement the full merge sort algorithm
   - Start with pseudocode before actual implementation
   - Trace through your implementation with a small example

5. Analyze the time and space complexity
   - Derive the recurrence relation for merge sort
   - Compare with other sorting algorithms you've learned

Hint: Pay special attention to how the merge operation works - this is where the actual "sorting" happens, after the array has been divided into single-element subarrays.

For resources, I recommend visualizations on algorithms.wtf or the chapter on merge sort in "Introduction to Algorithms" by Cormen et al."""
    
    prompt = f"""{system_prompt}

Example Input: {example_input}

Example Output: {example_output}

Now help with this question using a similar educational approach. Use the following reference materials:

Reference Materials:
{context}

Student Question: {query}

Your Educational Guidance:"""
    
    try:
        import torch
        
        # Use GPU optimizations
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Generate response using DeepSeek
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                # Add performance optimizations
                use_cache=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the model's response (after the prompt)
            response = response.split("Your Educational Guidance:")[-1].strip()
            
            # Clear CUDA cache to free up memory
            torch.cuda.empty_cache()
            
            logger.info(f"Generated RAG response of length {len(response)}")
            return response
            
    except Exception as e:
        logger.error(f"Error during RAG inference: {e}")
        # Fall back to simpler generation if optimization fails
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=1024)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("Your Educational Guidance:")[-1].strip()
        return response
    
# Main function to process PDF and generate dataset - saving in smaller batches
def process_pdf_to_qa(
    pdf_path: str,
    api_key: str,
    chroma_client: chromadb.PersistentClient
) -> List[Dict[str, str]]:
    """Process a PDF into Q&A pairs and add to Chroma DB."""
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Setup Claude client
    client = setup_claude_client(api_key)
    
    # Extract text from PDF and get page count
    text, page_count = extract_text_from_pdf(pdf_path)
    
    # Calculate target number of Q&A pairs based on page count
    target_qa_pairs = page_count * QUESTIONS_PER_PAGE
    logger.info(f"Targeting {target_qa_pairs} Q&A pairs for {page_count} pages")
    
    # Split text into chunks
    chunks = split_text_into_chunks(text)
    
    # Calculate questions per chunk to reach target
    questions_per_chunk = max(1, min(5, (target_qa_pairs // len(chunks)) + 1))
    logger.info(f"Will generate approximately {questions_per_chunk} questions per chunk")
    
    # Generate dataset
    dataset = []
    pdf_name = os.path.basename(pdf_path)
    
    # Process in batches of 5 chunks at a time
    BATCH_SAVE_SIZE = 50
    
    for i, chunk in enumerate(chunks):
        if len(dataset) >= target_qa_pairs:
            break
            
        # Extract topic from chunk
        topic = extract_topic(chunk)
        
        # Generate questions from chunk
        questions = generate_questions_from_chunk(
            client, 
            chunk, 
            topic, 
            num_questions=questions_per_chunk
        )
        
        # For each question, generate a roadmap answer
        for question in questions:
            if len(dataset) >= target_qa_pairs:
                break
                
            answer = generate_roadmap_answer(client, question, chunk)
            
            # Add to dataset
            dataset.append({
                "input": question,
                "output": answer
            })
            
            logger.info(f"Generated datapoint {len(dataset)}/{target_qa_pairs}")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        logger.info(f"Processed chunk {i+1}/{len(chunks)}")
        
        # Save in smaller batches periodically
        if len(dataset) >= BATCH_SAVE_SIZE or i == len(chunks) - 1:
            # Add the generated Q&A pairs to Chroma DB
            if dataset:  # Only if we have data to add
                add_qa_to_chroma(chroma_client, dataset, pdf_name)
                logger.info(f"Added batch of {len(dataset)} Q&A pairs to Chroma DB")
            
            # Clear the dataset to start fresh for the next batch
            dataset = []
    
    logger.info(f"Processing complete for {pdf_path}")
    
    # Return the count of processed items
    collection = chroma_client.get_collection("cs_educational_qa")
    count = collection.count()
    logger.info(f"Chroma DB now contains {count} total Q&A pairs")
    
    return {"processed_pdf": pdf_name, "total_qa_pairs": count}
# Main application function
def main():
    parser = argparse.ArgumentParser(description="Educational RAG System for Computer Science")
    parser.add_argument("--pdf", help="Path to the PDF to process")
    parser.add_argument("--query", help="Query to the RAG system")
    parser.add_argument("--api-key", help="Claude API key")
    
    args = parser.parse_args()
    
    # Set up Chroma DB
    chroma_client = setup_chroma_db()
    
    # Set up DeepSeek model if we're going to query
    if args.query:
        model, tokenizer = setup_deepseek_model()
    
    # Process PDF if provided
    if args.pdf:
        process_pdf_to_qa(args.pdf, args.api_key, chroma_client)
    
    # Query the system if a query is provided
    if args.query:
        response = query_rag_system(args.query, chroma_client, model, tokenizer)
        print("\n" + "="*80)
        print("STUDENT QUERY:", args.query)
        print("-"*80)
        print("EDUCATIONAL GUIDANCE:")
        print(response)
        print("="*80)

if __name__ == "__main__":
    main()