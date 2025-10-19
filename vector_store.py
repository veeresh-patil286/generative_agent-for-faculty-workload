try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAS_FAISS = False
import numpy as np
import pickle
import os
from typing import List, Dict
import re

class PolicyVectorStore:
    """Vector store for university policies using FAISS, with NumPy fallback."""
    
    def __init__(self, persist_directory: str = "./faiss_db", collection_name: str = "university_policies"):
        """Initialize vector index and storage (FAISS if available, else NumPy)."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.dimension = 384  # Default embedding dimension
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Paths
        self.index_path = os.path.join(persist_directory, f"{collection_name}.index")
        self.metadata_path = os.path.join(persist_directory, f"{collection_name}_metadata.pkl")
        self.embeddings_path = os.path.join(persist_directory, f"{collection_name}_embeddings.npy")

        # Load existing data or initialize
        self.metadata = []
        self.embeddings = None  # Only used in NumPy fallback

        if HAS_FAISS:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        else:
            # NumPy fallback: load embeddings and metadata if present
            if os.path.exists(self.embeddings_path) and os.path.exists(self.metadata_path):
                self.embeddings = np.load(self.embeddings_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.embeddings = np.zeros((0, self.dimension), dtype='float32')
    
    def load_policies_from_file(self, policies_file: str = "policies.txt", force: bool = False):
        """Load policies from text file and store in vector database.

        If the collection is already populated, this is a no-op unless force=True.
        """
        try:
            # Avoid duplicating data on repeated runs unless forced
            if self.metadata and not force:
                return True
            with open(policies_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split policies into chunks (each policy rule is a chunk)
            policy_chunks = self._split_policies(content)
            
            # Prepare documents for embedding
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(policy_chunks):
                documents.append(chunk['text'])
                metadatas.append({
                    "policy_number": chunk['policy_number'],
                    "category": chunk['category'],
                    "source": "policies.txt",
                    "text": chunk['text']
                })
            
            # Simple random embeddings (demo only)
            embeddings = np.random.rand(len(documents), self.dimension).astype('float32')

            # Normalize embeddings for cosine similarity
            self._normalize_inplace(embeddings)

            # Add to index
            if HAS_FAISS:
                self.index.add(embeddings)
            else:
                if self.embeddings is None or self.embeddings.size == 0:
                    self.embeddings = embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, embeddings])

            self.metadata.extend(metadatas)

            # Persist
            self._save_index()
            
            print(f"Successfully loaded {len(documents)} policy chunks into vector store")
            return True
            
        except Exception as e:
            print(f"Error loading policies: {e}")
            return False
    
    def _split_policies(self, content: str) -> List[Dict]:
        """Split policy text into meaningful chunks."""
        lines = content.strip().split('\n')
        chunks = []
        
        current_chunk = ""
        policy_number = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a policy number (e.g., "1.", "2.", etc.)
            if re.match(r'^\d+\.', line):
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'policy_number': policy_number,
                        'category': self._categorize_policy(current_chunk)
                    })
                
                # Start new chunk
                policy_number = int(line.split('.')[0])
                current_chunk = line
            else:
                # Add to current chunk
                current_chunk += " " + line
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'policy_number': policy_number,
                'category': self._categorize_policy(current_chunk)
            })
        
        return chunks
    
    def _categorize_policy(self, policy_text: str) -> str:
        """Categorize policy based on content."""
        text_lower = policy_text.lower()
        
        if any(word in text_lower for word in ['workload', 'hours', 'teaching']):
            return "workload_management"
        elif any(word in text_lower for word in ['schedule', 'time', 'slot', 'break']):
            return "scheduling"
        elif any(word in text_lower for word in ['department', 'distribute', 'staff']):
            return "department_management"
        elif any(word in text_lower for word in ['research', 'administrative', 'mentoring']):
            return "faculty_development"
        else:
            return "general"
    
    def search_policies(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant policies based on query."""
        try:
            # Create a simple query embedding (random for demo)
            query_embedding = np.random.rand(1, self.dimension).astype('float32')
            self._normalize_inplace(query_embedding)

            policy_results = []

            if HAS_FAISS:
                scores, indices = self.index.search(query_embedding, n_results)
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self.metadata):
                        policy_results.append({
                            'text': self.metadata[idx]['text'],
                            'metadata': self.metadata[idx],
                            'distance': float(score)
                        })
            else:
                if self.embeddings is None or len(self.metadata) == 0:
                    return []
                # Cosine similarity via dot product (embeddings already normalized)
                sims = (self.embeddings @ query_embedding.T).ravel()
                top_indices = np.argsort(-sims)[:n_results]
                for idx in top_indices:
                    if 0 <= idx < len(self.metadata):
                        policy_results.append({
                            'text': self.metadata[idx]['text'],
                            'metadata': self.metadata[idx],
                            'distance': float(sims[idx])
                        })

            return policy_results
            
        except Exception as e:
            print(f"Error searching policies: {e}")
            return []
    
    def get_policy_by_category(self, category: str) -> List[Dict]:
        """Get all policies in a specific category."""
        try:
            # Filter metadata by category
            category_policies = []
            for i, metadata in enumerate(self.metadata):
                if metadata.get('category') == category:
                    category_policies.append({
                        'text': metadata['text'],
                        'metadata': metadata,
                        'distance': 1.0  # No distance for category filter
                    })
            return category_policies
            
        except Exception as e:
            print(f"Error getting policies by category: {e}")
            return []
    
    def get_all_policies(self) -> List[Dict]:
        """Get all policies in the collection."""
        try:
            policies = []
            for i, metadata in enumerate(self.metadata):
                policies.append({
                    'text': metadata['text'],
                    'metadata': metadata,
                    'distance': None
                })
            
            return policies
            
        except Exception as e:
            print(f"Error getting all policies: {e}")
            return []
    
    def clear_collection(self):
        """Clear all policies from the collection."""
        try:
            if HAS_FAISS:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.embeddings = np.zeros((0, self.dimension), dtype='float32')
            self.metadata = []
            self._save_index()
            print("Collection cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            return {
                "name": self.collection_name,
                "document_count": len(self.metadata),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": f"Error getting collection info: {e}"}
    
    def _save_index(self):
        """Save the vector index/embeddings and metadata to disk."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            if HAS_FAISS:
                faiss.write_index(self.index, self.index_path)
            else:
                if self.embeddings is not None:
                    np.save(self.embeddings_path, self.embeddings)
        except Exception as e:
            print(f"Error saving index: {e}")

    @staticmethod
    def _normalize_inplace(mat: np.ndarray) -> None:
        """L2-normalize rows of a matrix in-place for cosine similarity."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

# Example usage and testing
if __name__ == "__main__":
    # Initialize vector store
    vector_store = PolicyVectorStore()
    
    # Load policies
    success = vector_store.load_policies_from_file("policies.txt")
    
    if success:
        # Test search
        print("\n=== Testing Policy Search ===")
        search_queries = [
            "maximum workload hours per week",
            "faculty scheduling rules",
            "department workload distribution"
        ]
        
        for query in search_queries:
            print(f"\nQuery: {query}")
            results = vector_store.search_policies(query, n_results=2)
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['text'][:100]}...")
                print(f"     Category: {result['metadata']['category']}")
        
        # Test category search
        print("\n=== Testing Category Search ===")
        categories = ["workload_management", "scheduling", "department_management"]
        for category in categories:
            policies = vector_store.get_policy_by_category(category)
            print(f"\n{category}: {len(policies)} policies")
            for policy in policies[:2]:  # Show first 2
                print(f"  - {policy['text'][:80]}...")
        
        # Collection info
        print("\n=== Collection Info ===")
        info = vector_store.get_collection_info()
        print(f"Collection: {info}")
