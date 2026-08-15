# cache.py

from functools import lru_cache


@lru_cache(maxsize=100)
def get_cached_query(query):
    """
    Stores results for repeated queries.
    """

    return None


def clear_cache():
    """
    Clear cache after documents are updated.
    """

    get_cached_query.cache_clear()