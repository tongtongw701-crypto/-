from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class ChromaV2Collection:
    id: str
    name: str
    count: int


class ChromaV2HttpClient:
    """
    最小可用的 Chroma Server v2 HTTP 客户端（专门用于查询检索）。
    目的：绕开某些环境下 chromadb.HttpClient 初始化时的异常包装/版本不兼容问题。
    """

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        ident = self.get_identity()
        self.tenant = ident.get("tenant", "default_tenant")
        dbs = ident.get("databases") or ["default_database"]
        self.database = dbs[0]

    def _get(self, path: str) -> requests.Response:
        r = requests.get(self.base_url + path, timeout=self.timeout)
        r.raise_for_status()
        return r

    def _post(self, path: str, json_body: Dict[str, Any]) -> requests.Response:
        r = requests.post(self.base_url + path, json=json_body, timeout=self.timeout)
        r.raise_for_status()
        return r

    def heartbeat(self) -> Dict[str, Any]:
        return self._get("/api/v2/heartbeat").json()

    def get_identity(self) -> Dict[str, Any]:
        r = self._get("/api/v2/auth/identity")
        return r.json()

    def list_collections(self) -> List[Dict[str, Any]]:
        r = self._get(f"/api/v2/tenants/{self.tenant}/databases/{self.database}/collections")
        return r.json()

    def count(self, collection_id: str) -> int:
        r = self._get(f"/api/v2/tenants/{self.tenant}/databases/{self.database}/collections/{collection_id}/count")
        return int(r.text.strip() or "0")

    def pick_collection(self, prefer_name: Optional[str] = "langchain") -> ChromaV2Collection:
        cols = self.list_collections()
        if not cols:
            raise RuntimeError("Chroma Server 未发现任何 collection。请确认 server 的 --path 指向正确的 vector_db。")

        if prefer_name:
            for c in cols:
                if c.get("name") == prefer_name:
                    cid = c["id"]
                    return ChromaV2Collection(id=cid, name=prefer_name, count=self.count(cid))

        # fallback: 最大 count
        best_id = cols[0]["id"]
        best_name = cols[0].get("name", "unknown")
        best_count = -1
        for c in cols:
            cid = c["id"]
            try:
                cnt = self.count(cid)
            except Exception:
                cnt = -1
            if cnt > best_count:
                best_id, best_name, best_count = cid, c.get("name", "unknown"), cnt
        return ChromaV2Collection(id=best_id, name=best_name, count=best_count)

    def query(
        self,
        collection_id: str,
        query_embeddings: List[List[float]],
        n_results: int = 4,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        include = include or ["documents", "metadatas", "distances"]
        payload = {
            "query_embeddings": query_embeddings,
            "n_results": int(n_results),
            "include": include,
        }
        r = self._post(
            f"/api/v2/tenants/{self.tenant}/databases/{self.database}/collections/{collection_id}/query",
            payload,
        )
        return r.json()


