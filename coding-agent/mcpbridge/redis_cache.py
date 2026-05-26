import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisEntityCache:
    def __init__(self, redis_url: str, sync_interval: int = 60, top_k: int = 15) -> None:
        self._redis_url = redis_url
        self._sync_interval = sync_interval
        self._top_k = top_k
        self._client: Any | None = None

    async def connect(self) -> bool:
        try:
            import redis.asyncio as aioredis  # type: ignore[import]
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._client.ping()
            return True
        except Exception as e:
            logger.warning("Redis unavailable: %s", e)
            self._client = None
            return False

    async def sync_flows(self, raw_flows: list[dict]) -> None:
        if not self._client or not raw_flows:
            return
        pipe = self._client.pipeline()
        ids: list[str] = []
        for flow in raw_flows:
            fid = str(flow.get("id", ""))
            if not fid:
                continue
            ids.append(fid)
            pipe.hset(f"lf:flow:{fid}", mapping={
                "name": flow.get("name", ""),
                "description": flow.get("description", "") or "",
                "folder_id": str(flow.get("folder_id", "") or ""),
                "updated_at": str(flow.get("updated_at", "") or ""),
            })
        if ids:
            pipe.delete("lf:flows:ids")
            pipe.sadd("lf:flows:ids", *ids)
        await pipe.execute()

    async def sync_starters(self, raw_starters: list[dict]) -> None:
        if not self._client or not raw_starters:
            return
        pipe = self._client.pipeline()
        ids: list[str] = []
        for s in raw_starters:
            sid = str(s.get("id", ""))
            if not sid:
                continue
            ids.append(sid)
            pipe.hset(f"lf:starter:{sid}", mapping={
                "name": s.get("name", ""),
                "description": s.get("description", "") or "",
            })
            pipe.set(f"lf:starter:data:{sid}", json.dumps(s))
        if ids:
            pipe.delete("lf:starters:ids")
            pipe.sadd("lf:starters:ids", *ids)
        await pipe.execute()

    async def search_flows(self, query: str, limit: int | None = None) -> list[dict]:
        if not self._client:
            return []
        limit = limit or self._top_k
        ids = await self._client.smembers("lf:flows:ids")
        q = query.lower()
        results: list[dict] = []
        for fid in ids:
            data = await self._client.hgetall(f"lf:flow:{fid}")
            if not data:
                continue
            name = data.get("name", "")
            desc = data.get("description", "")
            if q in name.lower() or q in desc.lower():
                results.append({"id": fid, "name": name, "description": desc})
        results.sort(key=lambda x: x["name"])
        return results[:limit]

    async def list_all_flows(self, limit: int = 50) -> list[dict]:
        if not self._client:
            return []
        ids = await self._client.smembers("lf:flows:ids")
        flows: list[dict] = []
        for fid in ids:
            data = await self._client.hgetall(f"lf:flow:{fid}")
            if data:
                flows.append({
                    "id": fid,
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "updated_at": data.get("updated_at", ""),
                })
        flows.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return flows[:limit]

    async def list_all_starters(self) -> list[dict]:
        if not self._client:
            return []
        ids = await self._client.smembers("lf:starters:ids")
        starters: list[dict] = []
        for sid in ids:
            raw = await self._client.get(f"lf:starter:data:{sid}")
            if raw:
                try:
                    starters.append(json.loads(raw))
                except Exception:
                    pass
        return starters

    async def is_warm(self) -> bool:
        if not self._client:
            return False
        try:
            count = await self._client.scard("lf:flows:ids")
            return count > 0
        except Exception:
            return False

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
