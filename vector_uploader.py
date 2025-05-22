'''
Processes a JSON file containing scraped website content, vectorizes the content, 
and uploads it to a specified vector database (Pinecone or Weaviate).
'''
import json
import os
import uuid
import time
import numpy as np
from sentence_transformers import CrossEncoder 

from dotenv import load_dotenv

load_dotenv()

# --- Environment Variable Loading ---
# pinecone_api_key = os.getenv("PINECONE_API_KEY") # Removed
weaviate_url = os.getenv("WEAVIATE_URL") # E.g., "https://your-cluster.weaviate.network"
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
# openai_api_key_for_embeddings = os.getenv("OPENAI_API_KEY_FOR_EMBEDDINGS") # REMOVED
google_api_key = os.getenv("GOOGLE_API_KEY") # Used if EMBEDDING_MODEL_TYPE is gemini with Google SDK
LOCAL_EMBEDDING_MODEL_PATH = "embedding_models/inf-retriever-v1-1.5b"

# --- Global Configuration ---
# Choose your vector database: "pinecone" or "weaviate"
# VECTOR_DB_TYPE = "pinecone"  # or "weaviate" # Removed - only Weaviate

# MODIFIED: Choose your embedding model type: "local" or "gemini"
EMBEDDING_MODEL_TYPE = "local" # Default to local, can be "gemini"

# Pinecone specific configuration
# PINECONE_INDEX_NAME = "web-content-index" # Choose a suitable name # Removed

# Weaviate specific configuration
WEAVIATE_CLASS_NAME_PREFIX = "WebContent" # Class name will be {PREFIX}ContentChunks

# --- Embedding Model Loading ---
model = None
EMBEDDING_DIMENSION = 0

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers library not found. Please install with: pip install sentence-transformers")
    # Exit if essential library is missing, or handle more gracefully
    # For now, we'll let it fail later if SentenceTransformer is needed and not loaded.

if EMBEDDING_MODEL_TYPE == "local": # MODIFIED
    try:
        if not os.path.isdir(LOCAL_EMBEDDING_MODEL_PATH):
            raise FileNotFoundError(f"Local embedding model path not found or not a directory: {LOCAL_EMBEDDING_MODEL_PATH}")
        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_PATH)
        EMBEDDING_DIMENSION = model.get_sentence_embedding_dimension()
        print(f"Loaded local SentenceTransformer model from: {LOCAL_EMBEDDING_MODEL_PATH}. Dimension: {EMBEDDING_DIMENSION}")
    except Exception as e:
        print(f"Error loading local SentenceTransformer model from '{LOCAL_EMBEDDING_MODEL_PATH}': {e}")
elif EMBEDDING_MODEL_TYPE == "gemini":
    EMBEDDING_DIMENSION = 768 # Common dimension for Gemini embeddings
    print(f"GEMINI model type selected. Dimension set to: {EMBEDDING_DIMENSION}")
    print("Ensure 'model.encode()' is correctly implemented for Gemini embeddings.")
    print("Example using Google Generative AI SDK (ensure GOOGLE_API_KEY is set):")
    print("  import google.generativeai as genai")
    print("  genai.configure(api_key=google_api_key)")
    print("  gemini_model_instance = genai.GenerativeModel('models/embedding-001')")
    print("  def gemini_encode_func(texts_list):")
    print("      # Note: Gemini API might take one text or a list depending on version/method")
    print("      # This is a simplified example for batching if model supports it, or loop if not.")
    print("      all_embeddings = []")
    print("      for text_input in texts_list:")
    print("          result = genai.embed_content(model='models/embedding-001', content=text_input, task_type=\"RETRIEVAL_DOCUMENT\")")
    print("          all_embeddings.append(result['embedding'])")
    print("      return np.array(all_embeddings)")
    print("  # Assign this function to model.encode or integrate directly:")
    print("  # model = type('obj', (object,), {'encode' : gemini_encode_func})()")
    # As a fallback placeholder if user doesn't set up their Gemini model:
    try:
        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # 768 dim # REMOVED Placeholder for Gemini
        print("Gemini model selected. User needs to ensure 'model.encode' is implemented as per instructions.")
        print("No placeholder model loaded by default for Gemini. 'model' will be None until user implements it.")
    except Exception as e:
        print(f"Error during Gemini setup (placeholder check): {e}")
else:
    raise ValueError(f"Unsupported EMBEDDING_MODEL_TYPE: {EMBEDDING_MODEL_TYPE}. Must be 'local' or 'gemini'. Or model loading failed.")

if model is None and EMBEDDING_MODEL_TYPE == "local": # MODIFIED: Only critical if 'local' and failed
    raise ValueError("Local embedding model could not be loaded. Please check configurations and dependencies.")
elif model is None and EMBEDDING_MODEL_TYPE == "gemini":
    print("Warning: Gemini embedding model is not initialized by this script. User implementation required.")


# --- Reranking Model Loading ---
reranking_model = None
try:
    # As per user example: tomaarsen/reranker-msmarco-ModernBERT-base-lambdaloss
    # Added trust_remote_code=True as in the user's example context
    reranking_model = CrossEncoder('reranker_models/reranker-msmarco-ModernBERT-base-lambdaloss')
    print("Successfully loaded Reranking Model: tomaarsen/reranker-msmarco-ModernBERT-base-lambdaloss")
except Exception as e:
    print(f"Error loading Reranking Model: {e}. Reranking functionality will be unavailable.")
    print("Ensure 'cross-encoders' (or 'rerankers') library is installed and the model name is correct.")

# --- Weaviate Imports and Functions ---
try:
    import weaviate
    import weaviate.classes.config as wvc
    from weaviate.auth import Auth
    from weaviate.classes.query import MetadataQuery # ADDED
    # from cross_encoders import CrossEncoder # Moved to top
except ImportError:
    # Combined ImportError messages
    print("Weaviate client, CrossEncoders, or other necessary libraries not found.")
    print("Please install with: pip install weaviate-client sentence-transformers cross-encoders")
    # Consider exiting or setting flags to disable functionalities

def initialize_weaviate_client_and_schema():
    """
    Initializes Weaviate client and ensures schema exists.
    Modifications:
    - Adds 'web_page_url' to schema.
    - Uses WEAVIATE_CLASS_NAME_PREFIX from global config.
    - REMOVED: Uses openai_api_key_for_embeddings for OpenAI header.
    """
    if not weaviate_url or not weaviate_api_key:
        print("Error: WEAVIATE_URL or WEAVIATE_API_KEY environment variable not set.")
        return None, None

    client = None
    additional_headers = {}
    # REMOVED: OpenAI specific header logic
    # if EMBEDDING_MODEL_TYPE == "openai" and openai_api_key_for_embeddings: 
    #     additional_headers["X-OpenAI-Api-Key"] = openai_api_key_for_embeddings
    #     print("Including OpenAI API Key in headers for Weaviate connection (for OpenAI vectorizer if used in schema).")

    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            headers=additional_headers if additional_headers else None
        )
        if not client.is_ready(): # is_ready() is the method to check connection
            print(f"Failed to connect to Weaviate Cloud at {weaviate_url}.")
            if client: client.close()
            return None, None
        print(f"Successfully connected to Weaviate Cloud: {weaviate_url}.")
    except Exception as e:
        print(f"Error connecting to Weaviate Cloud: {e}")
        if client: client.close()
        return None, None

    class_name = f"{WEAVIATE_CLASS_NAME_PREFIX}ContentChunks" # Use global prefix
    if not class_name[0].isupper(): class_name = class_name[0].upper() + class_name[1:]

    if not client.collections.exists(class_name):
        print(f"Weaviate class '{class_name}' does not exist. Creating now...")
        try:
            client.collections.create(
                name=class_name,
                vectorizer_config=wvc.Configure.Vectorizer.none(), # MODIFIED
                properties=[
                    wvc.Property(name="content", data_type=wvc.DataType.TEXT), # MODIFIED
                    wvc.Property(name="page_number", data_type=wvc.DataType.INT), # MODIFIED
                    wvc.Property(name="document", data_type=wvc.DataType.TEXT), # MODIFIED
                    wvc.Property(name="topic", data_type=wvc.DataType.TEXT), # MODIFIED
                    wvc.Property(name="web_page_url", data_type=wvc.DataType.TEXT), # MODIFIED
                    wvc.Property(name="chunk_id", data_type=wvc.DataType.TEXT), # MODIFIED
                ]
            )
            print(f"Successfully created Weaviate class: {class_name}")
        except Exception as e:
            print(f"Error creating class '{class_name}' in Weaviate: {e}")
            client.close()
            return None, None
    else:
        print(f"Weaviate class '{class_name}' already exists.")
    return client, class_name

def vectorize_n_upsert_vector_to_weaviate(
    json_path: str,
    client: 'weaviate.WeaviateClient', # Type hint for clarity
    class_name: str
):
    """
    Reads JSON, vectorizes content, and upserts to Weaviate.
    Modifications:
    - Uses item_index as chunk_id (string) if 'chunk_id' is not in item.
    - Uses meta_data.web_site_name for 'document' if 'document' is not in meta_data.
    - Stores meta_data.web_page_url.
    - chunk_id is now string.
    """
    if not client or not class_name:
        print("Weaviate client or class name not provided. Aborting."); return []
    if model is None:
        print("Embedding model is not loaded. Aborting upsert."); return []

    processed_chunk_ids = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content_items = json.load(f)
            if not isinstance(content_items, list):
                print(f"Error: JSON file at {json_path} does not contain a list of items."); return []
    except Exception as e:
        print(f"Error reading or parsing JSON file '{json_path}': {e}"); return []

    collection = client.collections.get(class_name)
    
    with collection.batch.dynamic() as batch:
        for item_index, item in enumerate(content_items):
            if not isinstance(item, dict):
                print(f"Skipping non-dictionary item at index {item_index}."); continue

            content_to_vectorize = item.get('content')
            if not content_to_vectorize or not isinstance(content_to_vectorize, str):
                print(f"Skipping item at index {item_index} due to missing/invalid 'content'."); continue
            
            # Use item_index as chunk_id if not present, ensure it's a string
            chunk_id_str = str(item.get('chunk_id', item_index))

            try:
                embedding_array = model.encode([content_to_vectorize])
                if embedding_array is None or embedding_array.ndim != 2 or embedding_array.shape[0] != 1:
                    print(f"Failed to vectorize content for item at index {item_index}."); continue
                vector = embedding_array[0].tolist()
            except Exception as e:
                print(f"Error vectorizing content for item at index {item_index}: {e}"); continue

            properties = {"content": content_to_vectorize, "chunk_id": chunk_id_str}
            meta_data_dict = item.get('meta_data', {})
            if not isinstance(meta_data_dict, dict): meta_data_dict = {}

            page_num = meta_data_dict.get('page_number', item.get('page_number'))
            if page_num is not None:
                try: properties["page_number"] = int(page_num)
                except (ValueError, TypeError): print(f"Warning: Could not convert page_number for chunk {chunk_id_str}.")
            
            properties["document"] = meta_data_dict.get('document', meta_data_dict.get('web_site_name', "N/A"))
            properties["topic"] = meta_data_dict.get('topic', "N/A")
            properties["web_page_url"] = meta_data_dict.get('web_page_url', "N/A")
            
            try:
                batch.add_object(properties=properties, vector=vector, uuid=str(uuid.uuid4()))
                processed_chunk_ids.append(chunk_id_str)
            except Exception as e:
                print(f"Error adding item for chunk_id {chunk_id_str} to Weaviate batch: {e}")

    if not processed_chunk_ids:
        print(f"No items were successfully added to Weaviate batch from '{json_path}'.")
    else:
        print(f"Successfully added {len(processed_chunk_ids)} items to Weaviate batch from '{json_path}'.")
    return processed_chunk_ids

def query_weaviate_for_similar_schemas(query_text, top_k=1):
    """Queries Weaviate for content similar to the query text."""
    client, class_name = initialize_weaviate_client_and_schema()
    if not client or not class_name: return []
    if model is None: print("Embedding model not loaded."); client.close(); return []

    try:
        embedding_array = model.encode([query_text])
        if embedding_array is None or embedding_array.ndim != 2 or embedding_array.shape[0] != 1:
            print(f"Failed to vectorize query: '{query_text}'."); client.close(); return []
        query_vector = embedding_array[0].tolist()
    except Exception as e:
        print(f"Error vectorizing query '{query_text}': {e}"); client.close(); return []

    try:
        collection = client.collections.get(class_name)
        response = collection.query.near_vector(
            near_vector=query_vector, limit=top_k,
            return_metadata=MetadataQuery(distance=True, certainty=True), # MODIFIED
            return_properties=True
        )
    except Exception as e:
        print(f"Error querying Weaviate: {e}"); client.close(); return []
    
    results = []
    if not response.objects:
        print("No matches found in Weaviate."); client.close(); return []

    for obj in response.objects:
        score = obj.metadata.certainty if obj.metadata.certainty is not None else (1.0 - obj.metadata.distance if obj.metadata.distance is not None else 0.0)
        results.append({"id": str(obj.uuid), "score": score, "metadata": obj.properties if obj.properties else {}})
    
    client.close()
    return results

# --- New Function for Querying and Reranking ---
def query_and_rerank_weaviate_results(query_text: str, initial_top_k: int = 10, rerank_top_y: int = 3):
    """
    Queries Weaviate, retrieves initial top_k results, reranks them using .rank(), 
    returns top_y results, and logs initial and reranked results to results.txt.
    """
    output_log_file = "results.txt"

    if model is None:
        print("Embedding model (for initial query) is not loaded. Aborting rerank.")
        return []
    if reranking_model is None:
        print("Reranking model is not loaded. Aborting rerank.")
        return []

    # 1. Generate embedding for the query text
    query_vector = None
    # latency_ms = 0 # Not used in the return of this simplified version
    try:
        # start_time = time.time() # For latency calculation if needed
        embedding_array = model.encode([query_text]) # Expects a list
        # latency_ms = (time.time() - start_time) * 1000
        if embedding_array is None or embedding_array.ndim != 2 or embedding_array.shape[0] != 1:
            print("Failed to encode the query or unexpected embedding shape for rerank.")
            return []
        query_vector = embedding_array[0].tolist()
    except Exception as e:
        print(f"Error vectorizing query '{query_text}' for rerank: {e}")
        return []

    # 2. Initialize Weaviate client and get class name (with retry)
    client = None
    class_name = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            temp_client, temp_class_name = initialize_weaviate_client_and_schema()
            # Check if client is connected properly
            if temp_client and temp_class_name and temp_client.is_connected() and temp_client.is_ready():
                client = temp_client
                class_name = temp_class_name
                break
            else:
                if temp_client: temp_client.close() # Close if partially opened but not ready
                client = None 
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} to init Weaviate client failed for rerank. Retrying...")
                    time.sleep(1) 
                else:
                    print("Maximum retries reached for Weaviate client initialization during rerank.")
        except Exception as e:
            print(f"Exception during Weaviate client init attempt {attempt + 1} for rerank: {e}")
            if client: client.close() # Ensure client from a failed attempt is closed
            client = None
            if attempt >= max_retries - 1:
                print("Maximum retries reached due to exception during Weaviate init for rerank.")
                break
    
    if not client or not class_name:
        print("Failed to initialize Weaviate client for rerank. Aborting.")
        return []

    # 3. Query Weaviate for initial_top_k results
    initial_results_raw = []
    try:
        collection = client.collections.get(class_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=initial_top_k,
            return_metadata=MetadataQuery(distance=True, certainty=True), # MODIFIED
            return_properties=True 
        )
        if response.objects:
            initial_results_raw = response.objects
    except Exception as e:
        print(f"Error querying Weaviate class '{class_name}' for rerank: {e}")
        client.close()
        return []

    if not initial_results_raw:
        print("No initial matches found in Weaviate for reranking.")
        client.close()
        return []

    # 4. Prepare for Reranking
    rerank_candidates = []
    for obj in initial_results_raw:
        if obj.properties and 'content' in obj.properties:
            initial_score = 0.0
            if obj.metadata.certainty is not None:
                initial_score = obj.metadata.certainty
            elif obj.metadata.distance is not None: # Lower distance is better
                initial_score = 1.0 - obj.metadata.distance 
            
            rerank_candidates.append({
                "id": str(obj.uuid),
                "initial_score": initial_score,
                "content_for_rerank": obj.properties['content'],
                "metadata": obj.properties 
            })
        # else:
            # print(f"Skipping result {obj.uuid} for reranking (missing 'content').") # Optional: too verbose

    if not rerank_candidates:
        print("No suitable candidates for reranking (e.g., all missing 'content').")
        if client: client.close()
        with open(output_log_file, 'a', encoding='utf-8') as f_log:
            f_log.write("\n--- No Suitable Candidates for Reranking ---\n")
            f_log.write("No candidates found with 'content' property for reranking.\n")
        return []

    # Log initial results (rerank_candidates already has the necessary info)
    with open(output_log_file, 'a', encoding='utf-8') as f_log:
        f_log.write(f"\n--- Initial Top {len(rerank_candidates)} Results for Query: '{query_text}' ---\n")
        for i, candidate in enumerate(rerank_candidates):
            f_log.write(f"\nRank {i+1} (Initial Retrieval):\n")
            f_log.write(f"  ID: {candidate.get('id', 'N/A')}\n")
            f_log.write(f"  Initial Score: {candidate.get('initial_score', 0.0):.4f}\n")
            f_log.write(f"  Content: {candidate.get('content_for_rerank', 'N/A')}\n")
            f_log.write(f"  Metadata: {json.dumps(candidate.get('metadata', {}), indent=2)}\n")
            f_log.write("-" * 20 + "\n")

    # 5. Perform Reranking using reranking_model.rank()
    documents_for_reranking = [candidate['content_for_rerank'] for candidate in rerank_candidates]
    
    final_results = [] 
    try:
        # reranking_model.rank returns a list of dicts, e.g., [{'corpus_id': idx, 'score': float, 'text': str}, ...]
        # It's sorted by score if top_k is used. Using rerank_top_y as top_k.
        # The 'text' key contains the document if return_documents=True.
        raw_rerank_output = reranking_model.rank(
            query=query_text,
            documents=documents_for_reranking,
            return_documents=True, # As per example, ensures 'text' and 'corpus_id' are in results
            top_k=rerank_top_y 
        )
        
        # Map reranked results (which are based on corpus_id) back to original candidate info
        for rerank_item in raw_rerank_output:
            original_candidate_index = rerank_item.get('corpus_id') # Use .get for safety
            rerank_score = rerank_item.get('score')

            if original_candidate_index is not None and rerank_score is not None:
                if 0 <= original_candidate_index < len(rerank_candidates):
                    # Get the full candidate dictionary and add/update rerank_score
                    full_candidate_info = rerank_candidates[original_candidate_index].copy()
                    full_candidate_info['rerank_score'] = rerank_score
                    full_candidate_info['original_initial_rank'] = original_candidate_index + 1 # Store original rank
                    # 'text' from rerank_item is documents_for_reranking[original_candidate_index]
                    # We already have this and more in full_candidate_info['metadata']['content']
                    final_results.append(full_candidate_info)
                else:
                    print(f"Warning: Invalid corpus_id {original_candidate_index} from reranker.")
            else:
                print(f"Warning: Rerank item missing 'corpus_id' or 'score': {rerank_item}")
            
    except AttributeError:
        print(f"Error: 'reranking_model' does not have a '.rank()' method. Ensure the correct CrossEncoder library (e.g., 'rerankers') is used and imported.")
        if client: client.close() 
        return []
    except Exception as e:
        print(f"Error during reranking model prediction with .rank(): {e}")
        if client: client.close() 
        return []

    # final_results is already sorted and truncated by reranking_model.rank
    
    # Log reranked results
    with open(output_log_file, 'a', encoding='utf-8') as f_log:
        if final_results:
            f_log.write(f"\\n\\n--- Top {len(final_results)} Reranked Results (out of initial {len(rerank_candidates)}) for Query: '{query_text}' ---\\n")
            for i, item in enumerate(final_results):
                f_log.write(f"\\nRank {i+1} (Reranked):\\n")
                f_log.write(f"  ID: {item.get('id', 'N/A')}\\n")
                f_log.write(f"  Rerank Score: {item.get('rerank_score', 0.0):.4f}\\n")
                f_log.write(f"  Original Initial Rank: {item.get('original_initial_rank', 'N/A')}\\n") # Log original rank
                initial_score_display = f"{item.get('initial_score', 0.0):.4f}" if 'initial_score' in item else "N/A"
                f_log.write(f"  Original Initial Score: {initial_score_display}\\n")
                f_log.write(f"  Content: {item.get('metadata', {}).get('content', 'N/A')}\\n")
                f_log.write(f"  Full Metadata: {json.dumps(item.get('metadata', {}), indent=2)}\\n")
                f_log.write("-" * 20 + "\\n")
        else:
            f_log.write("\\n--- No Reranked Results ---\\n")
            f_log.write("Reranking process did not yield any results (or an error occurred).\n")

    if client: client.close()
    return final_results

# --- Main Processing Function ---
def process_json_and_upload(json_file_path: str):
    """
    Main function to process the JSON file and upload to the configured vector database.
    """
    print(f"Starting processing for file: {json_file_path}")
    # print(f"Target Vector DB: {VECTOR_DB_TYPE}") # Removed
    print(f"Target Vector DB: Weaviate")
    print(f"Embedding Model Type: {EMBEDDING_MODEL_TYPE}")

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    # if VECTOR_DB_TYPE == "pinecone": # Removed
    #     print("Processing for Pinecone...")
    #     vectorize_n_upsert_vector_to_pinecone_db(json_file_path)
    # elif VECTOR_DB_TYPE == "weaviate": # Now only Weaviate
    print("Processing for Weaviate...")
    client, class_name = initialize_weaviate_client_and_schema()
    if client and class_name:
        try:
            vectorize_n_upsert_vector_to_weaviate(json_file_path, client, class_name)
        finally:
            client.close()
            print("Weaviate client closed.")
    # else: # Removed
    #     print(f"Unsupported VECTOR_DB_TYPE: {VECTOR_DB_TYPE}")

if __name__ == "__main__":

    # --- Configuration Check ---
    print("--- Configuration Summary ---")
    # print(f"  VECTOR_DB_TYPE: {VECTOR_DB_TYPE}") # Removed
    print(f"  Target Vector DB: Weaviate")
    print(f"  EMBEDDING_MODEL_TYPE: {EMBEDDING_MODEL_TYPE}")
    # if VECTOR_DB_TYPE == "pinecone": # Removed
    #     print(f"  PINECONE_API_KEY Set: {'Yes' if pinecone_api_key else 'No - REQUIRED'}")
    #     print(f"  PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")
    # elif VECTOR_DB_TYPE == "weaviate": # Now only Weaviate
    print(f"  WEAVIATE_URL Set: {'Yes' if weaviate_url else 'No - REQUIRED'}")
    print(f"  WEAVIATE_API_KEY Set: {'Yes' if weaviate_api_key else 'No - REQUIRED'}")
    print(f"  WEAVIATE_CLASS_NAME_PREFIX: {WEAVIATE_CLASS_NAME_PREFIX}")
    if EMBEDDING_MODEL_TYPE == "local": # MODIFIED
        print(f"  LOCAL_EMBEDDING_MODEL_PATH: {LOCAL_EMBEDDING_MODEL_PATH} (Exists: {os.path.isdir(LOCAL_EMBEDDING_MODEL_PATH)})")
    # REMOVED: OpenAI API key check
    # if EMBEDDING_MODEL_TYPE == "openai":
    #     print(f"  OPENAI_API_KEY_FOR_EMBEDDINGS Set: {'Yes' if openai_api_key_for_embeddings else 'No - REQUIRED for OpenAI embeddings'}")
    if EMBEDDING_MODEL_TYPE == "gemini":
         print(f"  GOOGLE_API_KEY Set: {'Yes' if google_api_key else 'No - REQUIRED for Google Gemini SDK'}")
    print("---------------------------")

    # Path to the JSON file from your scraper
    # Replace this with the actual path to your scraper's output JSON file.
    input_json_file = "test_web_data.json" 
    # Example: input_json_file = "c_Users_raninduh_Documents_Web_scraper_example_com.json"
    
    # if model is None and EMBEDDING_MODEL_TYPE == "gemini": # MODIFIED
    #     print(f"Warning: Embedding model for '{EMBEDDING_MODEL_TYPE}' is not fully initialized in this script.")
    #     print("Please ensure you have integrated your custom model loading and encode function as per the instructions.")
    #     # print("Attempting to proceed with placeholder if one was loaded, or it will fail if model is still None.") # Placeholder removed for Gemini
    # elif model is None and EMBEDDING_MODEL_TYPE == "local": # MODIFIED
    #     print("Critical Error: Local embedding model is None. Cannot proceed.")
    # else:
    #     # process_json_and_upload(input_json_file)

    #     # Example Query (after upserting)
    #     print("\\n--- Example Query ---")
    #     query = "Whate are the Patronus evaluation tools designed to do?"


    #     results = query_weaviate_for_similar_schemas(query, top_k=1)
    #     print(f"Weaviate query results for '{query}': {results}")

    # --- Example Reranking Query ---
    print("\\n--- Example Reranking Query ---")
    rerank_query = "Whate are the Patronus evaluation tools designed to do" # Changed query slightly for variety
    
    # Clear/Create results.txt for the new query session
    with open("results.txt", 'w', encoding='utf-8') as f_clear:
        f_clear.write(f"Query Log for: {rerank_query}\n")
        f_clear.write(f"Timestamp: {time.asctime()}\n")

    if model and reranking_model: 
        print(f"Performing reranking query for: '{rerank_query}'")
        # Ensure data is in Weaviate. If using dummy_json_path, it should be processed first.
        # process_json_and_upload(input_json_file) # Make sure this has run if DB is empty
        
        # Example: Get 5 initial results, rerank and return top 2
        reranked_items = query_and_rerank_weaviate_results(rerank_query, initial_top_k=10, rerank_top_y=4)
        
        if reranked_items:
            print(f"Top {len(reranked_items)} reranked results for '{rerank_query}':")
            for i, item in enumerate(reranked_items):
                print(f"  Rank {i+1}:")
                print(f"    ID: {item['id']}")
                print(f"    Rerank Score: {item['rerank_score']:.4f}")
                # Ensure initial_score exists before formatting
                initial_score_display = f"{item['initial_score']:.4f}" if 'initial_score' in item else "N/A"
                print(f"    Initial Score: {initial_score_display}")
                content_display = item['metadata'].get('content', "N/A")
                print(f"    Content: {content_display[:100]}...")
                print("-" * 20)
        else:
            print(f"No reranked results found for '{rerank_query}'.")
    else:
        print("Skipping reranking example: embedding or reranking model not loaded.")
