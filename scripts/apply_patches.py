import logging
import sys
import types
from functools import wraps
import inspect

logger = logging.getLogger(__name__)

def apply_patches():
    """
    Applies runtime patches to graphiti_core to fix node label issues.
    Replaces 'SET n:$(node.labels)' with 'SET n:Entity' in generated Cypher queries.
    This replaces the hardcoded patching in Dockerfile.
    """
    # Check if graphiti_core is installed/imported
    if 'graphiti_core' not in sys.modules:
        try:
            import graphiti_core
            import graphiti_core.utils.bulk_utils
            import graphiti_core.models.nodes.node_db_queries
        except ImportError:
            logger.warning("⚠️ graphiti_core not found. Skipping patches.")
            return

    logger.info("🔧 Applying runtime patches to graphiti_core...")
    patches_applied = 0

    # --- Patch 1: bulk_utils.bulk_import_statement_for_node ---
    try:
        from graphiti_core.utils import bulk_utils
        
        if hasattr(bulk_utils, 'bulk_import_statement_for_node'):
            original_bulk = bulk_utils.bulk_import_statement_for_node
            
            # Check if already patched to avoid recursion loop
            if not getattr(original_bulk, '_is_patched', False):
                @wraps(original_bulk)
                def patched_bulk_import(node_type, properties, id_property):
                    statement = original_bulk(node_type, properties, id_property)
                    if isinstance(statement, str) and "SET n:$(node.labels)" in statement:
                        # logger.debug(f"Patching bulk import statement for {node_type}")
                        return statement.replace("SET n:$(node.labels)", f"SET n:Entity")
                    return statement
                
                patched_bulk_import._is_patched = True
                bulk_utils.bulk_import_statement_for_node = patched_bulk_import
                patches_applied += 1
                logger.info("✅ Patched bulk_utils.bulk_import_statement_for_node")
    except Exception as e:
        logger.error(f"❌ Failed to patch bulk_utils: {e}")

    # --- Patch 2: models.nodes.node_db_queries (Generic search for problematic strings) ---
    try:
        from graphiti_core.models.nodes import node_db_queries
        
        # We look for functions that might return the problematic string
        for name, obj in inspect.getmembers(node_db_queries):
            if inspect.isfunction(obj):
                # We can't easily check the source code of compiled/installed modules in all envs,
                # but we can wrap functions that return strings.
                # Heuristic: verify if it's likely a query generator
                if "query" in name.lower() or "statement" in name.lower() or "string" in name.lower():
                    original_func = obj
                    if getattr(original_func, '_is_patched', False):
                        continue

                    @wraps(original_func)
                    def patched_query_func(*args, **kwargs):
                        result = original_func(*args, **kwargs)
                        if isinstance(result, str) and "SET n:$(node.labels)" in result:
                            # logger.debug(f"Patching query in {name}")
                            return result.replace("SET n:$(node.labels)", "SET n:Entity")
                        return result
                    
                    patched_query_func._is_patched = True
                    setattr(node_db_queries, name, patched_query_func)
                    # Just count unique functions patched
                    patches_applied += 1
        
        if patches_applied > 1:
             logger.info("✅ Patched node_db_queries methods")

    except Exception as e:
        logger.error(f"❌ Failed to patch node_db_queries: {e}")

    if patches_applied == 0:
        logger.warning("⚠️ No patches were applied. Check if graphiti_core version matches expectations (0.23.0).")
