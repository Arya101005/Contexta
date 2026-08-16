from functools import lru_cache  # built-in Python LRU cache decorator


@lru_cache(maxsize=100)  # cache up to 100 recent query results
def get_cached_query(query):
    """Cache for repeated queries — currently a stub, returns None."""
    return None


def clear_cache():
    """Clear the query cache — call after documents are updated."""
    get_cached_query.cache_clear()
