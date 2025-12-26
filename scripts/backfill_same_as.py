import os
import sys
import logging
import asyncio
from typing import Optional
import re

# Add parent dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backfill_same_as")

load_dotenv()

STOP_WORDS = {
    "project", "system", "data", "memory", "graph", "ai", "model", 
    "user", "assistant", "chat", "summary", "context", "fact",
    "проект", "система", "данные", "память", "граф", "ии", "модель", 
    "пользователь", "ассистент", "чат", "саммари", "контекст", "факт",
    "unknown", "none", "null"
}

def normalize_name(name: str) -> Optional[str]:
    if not name:
        return None
    
    # 1. Lowercase and trim
    norm = name.lower().strip()
    
    # 2. Cyrillic normalization
    norm = norm.replace('ё', 'е')
    
    # 3. Remove punctuation (keep alphanumeric and spaces)
    norm = re.sub(r'[^\w\s]', '', norm)
    
    # 4. Collapse whitespace
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    # 5. Check length and stop words
    if len(norm) < 3:  # Requirement said >= 4 or 3 with checks. Let's start with 3.
        return None
        
    if norm in STOP_WORDS:
        return None
        
    return norm

def run_backfill(use_embedding: bool = False):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "please_change_me")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        logger.info("Starting Backfill: SAME_AS Bridges")
        
        # 1. Normalize Names
        logger.info("--- Step 1: Normalizing Names ---")
        # Fetch all entities
        result = session.run("MATCH (e:Entity) WHERE e.name IS NOT NULL RETURN e.uuid as uuid, e.name as name")
        updates = []
        count_norm = 0
        
        for record in result:
            norm = normalize_name(record['name'])
            if norm:
                updates.append({"uuid": record['uuid'], "name_norm": norm})
                count_norm += 1
        
        if updates:
            # Batch update (simplified for script, can be chunked for huge DBs)
            logger.info(f"Updating {len(updates)} entities with name_norm...")
            session.run("""
                UNWIND $updates as up
                MATCH (e:Entity {uuid: up.uuid})
                SET e.name_norm = up.name_norm
            """, updates=updates)
        else:
            logger.info("No entities to normalize.")

        # 2. Create Exact Bridges
        logger.info("--- Step 2: Creating Exact Match Bridges ---")
        res = session.run("""
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1.name_norm IS NOT NULL
              AND e1.name_norm = e2.name_norm
              AND e1.uuid < e2.uuid
              AND e1.group_id <> e2.group_id
            MERGE (e1)-[r:SAME_AS]->(e2)
            RETURN count(r) as created
        """)
        created = res.single()['created']
        logger.info(f"Exact Match: Processed. Created/Merged {created} SAME_AS relationships.")
        
        # 3. Embedding Match (Optional)
        if use_embedding:
            logger.info("--- Step 3: Creating Embedding Match Bridges (Experimental) ---")
            # This requires vector index or manual cosine similarity. 
            # Doing Cartesian product on embedding similarity in Cypher is SLOW O(N^2).
            # We will rely on an index or skip if not scalable.
            # Assuming 'name_embedding' is a vector property.
            # Neo4j 5.x vector index query is best.
            
            # Check if index exists or use kNN. 
            # For backfill script, we might just iterate entities and query for neighbors.
            
            # Simplified approach: for each entity, find neighbors with similarity > 0.93
            # and different group_id. This is slow for many entities, but okay for < 10k.
            
            # Let's verify we have embeddings first
            has_embeddings = session.run("MATCH (e:Entity) WHERE e.name_embedding IS NOT NULL RETURN count(e) > 0 as has").single()['has']
            
            if has_embeddings:
                # We'll use a threshold based approach. 
                # Warning: This query is O(N^2) without a vector index. 
                # Better to use a dedicated Vector Index search if available.
                # Since we want to avoid APOC and complex index setup in this script, 
                # and assuming dataset is small (<1000 entities), we try a pure Cypher approach 
                # using vector.similarity.cosine if available.
                
                query_embedding = """
                MATCH (e1:Entity), (e2:Entity)
                WHERE e1.uuid < e2.uuid
                  AND e1.group_id <> e2.group_id
                  AND e1.name_embedding IS NOT NULL
                  AND e2.name_embedding IS NOT NULL
                  AND e1.name_norm IS NOT NULL 
                  AND size(e1.name_norm) >= 5 // Length constraint
                  AND size(e2.name_norm) >= 5
                  AND NOT (e1)-[:SAME_AS]-(e2)
                  
                WITH e1, e2, vector.similarity.cosine(e1.name_embedding, e2.name_embedding) as sim
                WHERE sim >= 0.93
                
                MERGE (e1)-[r:SAME_AS]->(e2)
                RETURN count(r) as created, avg(sim) as avg_sim
                """
                try:
                    res_emb = session.run(query_embedding)
                    rec = res_emb.single()
                    logger.info(f"Embedding Match: Created {rec['created']} bridges. Avg Sim: {rec['avg_sim']}")
                except Exception as e:
                    logger.warning(f"Embedding match failed (possibly vector functions not supported or memory issue): {e}")
            else:
                logger.info("No embeddings found on entities. Skipping Step 3.")

        # Summary
        res_summary = session.run("MATCH ()-[r:SAME_AS]->() RETURN count(r) as total")
        logger.info(f"Total SAME_AS bridges in graph: {res_summary.single()['total']}")

    driver.close()

if __name__ == "__main__":
    use_emb = "--embedding" in sys.argv
    run_backfill(use_embedding=use_emb)
