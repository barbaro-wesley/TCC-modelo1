"""Contadores Redis atomicos, compartilhados entre todos os workers."""

import hashlib

from fastapi import HTTPException
from redis import Redis
from redis.exceptions import RedisError

LUA = """
local count = redis.call('GET', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])
if count and tonumber(count) >= tonumber(ARGV[1]) then
    return {0, 0, math.max(ttl, 1)}
end
count = redis.call('INCR', KEYS[1])
if count == 1 or ttl < 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end
return {1, math.max(0, tonumber(ARGV[1]) - count), redis.call('TTL', KEYS[1])}
"""


class RateLimiter:
    def __init__(self, redis: Redis, prefix: str):
        self.redis, self.prefix = redis, prefix
        self.script = redis.register_script(LUA)

    def check(self, scope: str, identity: str, limit: int):
        key = hashlib.sha256(identity.encode()).hexdigest()
        try:
            allowed, remaining, ttl = self.script(
                keys=[f"{self.prefix}:{scope}:{key}"], args=[limit, 60]
            )
        except RedisError as exc:
            raise HTTPException(503, detail={"code": "rate_limiter_unavailable"}) from exc
        headers = {
            "RateLimit-Limit": str(limit),
            "RateLimit-Remaining": str(remaining),
            "RateLimit-Reset": str(ttl),
            "X-RateLimit-Scope": scope,
        }
        if not allowed:
            raise HTTPException(
                429,
                detail={"code": "rate_limit_exceeded", "scope": scope},
                headers={**headers, "Retry-After": str(ttl)},
            )
        return headers
