from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()  # Load the .env file
api_key = os.getenv("API_KEY")  # Fetch the key

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text).tolist()

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name="event_collection")

def chromaAdd(text, context):

  existing_docs = collection.count()
  # Generate the next ID
  next_id = str(existing_docs + 1)

  # Generate embeddings and store them
  collection.add(
      ids = [next_id],  # Unique identifierss
      documents = [text],
      embeddings = [get_embedding(text)],  # Generate embeddings
      metadatas = [{"source": context}]  # Add metadata
  )
  print("Documents added successfully!")


def chromaUpdate(text, doc_ID):
  collection.delete(ids = [doc_ID])
  collection.add(
      ids = [doc_ID],  # Unique identifierss
      documents = [text],
      embeddings = [get_embedding(text)],  # Generate embeddings
      metadatas = [{"source": "article"}]  # Add metadata
  )


def chromaQuery(text):
  query_embedding = get_embedding(text)

  # Retrieve the closest matches from ChromaDB
  results = collection.query(
      query_embeddings=[query_embedding],
      n_results=2  # Top 2 matches
  )
  return results


def rdRetrieval(query_text):
    results = chromaQuery(query_text)
    for doc, score in zip(results["documents"][0], results["distances"][0]):
        return {doc}

def rdPromptFormat(doc, query):
    return f'''Context = [{doc}]

    Query = {query}

    Using only the Context provided, answer the query. Do NOT mention about the CONTEXT.REMOVE ALL unnecessary factors like "./n",".\n", quotes etc from your reply.  START RIGHT WITH THE ANSWER. DO NOT ASK FOR SUGGESTIONS OR ANYTHING AT THE END.
    '''

client = genai.Client(api_key=api_key)
def rdLLM(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def rdRAG(user_prompt):

    document = rdRetrieval(user_prompt)
    query = rdPromptFormat(document, user_prompt)
    return rdLLM(query)