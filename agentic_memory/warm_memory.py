import json
from redis_client import redis_client


TTL_SECONDS = 3600


def write_warm_memory(user_id: str, data: dict):

    key = f"user:{user_id}:warm"

    redis_client.setex(
        key,
        TTL_SECONDS,
        json.dumps(data)
    )
    
    
    
def load_warm_memory(user_id: str):

    key = f"user:{user_id}:warm"

    data = redis_client.get(key)

    if not data:
        return {}

    return json.loads(data)


