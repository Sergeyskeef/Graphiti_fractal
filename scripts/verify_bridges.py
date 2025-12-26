from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "please_change_me")

driver = GraphDatabase.driver(uri, auth=(user, password))

def verify_bridges():
    with driver.session() as session:
        print("\n--- 3.1 User -> Episodes -> Entities (Sanity Check) ---")
        result1 = session.run("""
            MATCH (u:User)-[:AUTHORED]->(ep:Episodic)
            OPTIONAL MATCH (ep)-[:MENTIONS]->(e:Entity)
            RETURN count(u) as users, count(ep) as episodes, count(e) as entities
        """)
        print(result1.single().data())

        print("\n--- 3.2 Cross-Layer Bridges (SAME_AS) ---")
        result2 = session.run("""
            MATCH (e1:Entity)-[r:SAME_AS]->(e2:Entity)
            RETURN e1.name as name1, e1.group_id as group1, e2.name as name2, e2.group_id as group2
            LIMIT 50
        """)
        records = list(result2)
        print(f"Found {len(records)} SAME_AS connections.")
        for r in records:
            print(f"{r['name1']}({r['group1']}) <-> {r['name2']}({r['group2']})")

    driver.close()

if __name__ == "__main__":
    verify_bridges()
