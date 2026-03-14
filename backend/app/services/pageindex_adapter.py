"""Mock-first PageIndex adapter with local cache files."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.app.services.text_match import token_score


class PageIndexAdapter:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.documents: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def ingest_documents(self, base_id: str, documents: list[dict[str, Any]]) -> int:
        ingested = 0
        for document in documents:
            tree = self._build_tree(document)
            payload = {
                "document_id": document["id"],
                "title": document["title"],
                "source_path": document.get("source_path"),
                "tree": tree,
            }
            self.documents[base_id].append(payload)
            cache_path = self.cache_dir / base_id
            cache_path.mkdir(parents=True, exist_ok=True)
            with (cache_path / f"{document['id']}.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            ingested += 1
        return ingested

    def _build_tree(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        content = document.get("content") or ""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        nodes = []
        node_id = 1
        for line in lines:
            title = line.lstrip("# ").strip()
            if line.startswith("#"):
                nodes.append(
                    {
                        "node_id": f"node_{node_id}",
                        "title": title,
                        "summary": title,
                        "start_index": node_id,
                        "end_index": node_id,
                    }
                )
                node_id += 1
        if not nodes:
            nodes.append(
                {
                    "node_id": "node_1",
                    "title": document["title"],
                    "summary": (content[:120] or document["title"]).strip(),
                    "start_index": 1,
                    "end_index": 1,
                }
            )
        return nodes

    def query(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        filters = filters or {}
        base_id = filters.get("base_id")
        candidate_base_ids = [base_id] if base_id else list(self.documents.keys())
        hits: list[dict[str, Any]] = []
        selected_nodes: list[dict[str, Any]] = []
        for candidate_base_id in candidate_base_ids:
            for document in self.documents.get(candidate_base_id, []):
                for node in document.get("tree", []):
                    score = token_score(query, f"{node.get('title', '')} {node.get('summary', '')}")
                    if score <= 0:
                        continue
                    selected_nodes.append(
                        {
                            "document_id": document["document_id"],
                            "node_id": node["node_id"],
                            "title": node["title"],
                            "page_range": [node["start_index"], node["end_index"]],
                        }
                    )
                    hits.append(
                        {
                            "id": f"{document['document_id']}::{node['node_id']}",
                            "text": node["summary"],
                            "score": score,
                            "source": "pageindex",
                            "pageindex_node_id": node["node_id"],
                            "pageindex_title": node["title"],
                            "pageindex_page_range": [node["start_index"], node["end_index"]],
                            "metadata": {
                                "document_id": document["document_id"],
                                "base_id": candidate_base_id,
                                "pageindex_reasoning_selected": True,
                            },
                        }
                    )
        hits = sorted(hits, key=lambda item: item["score"], reverse=True)[:top_k]
        return {
            "results": deepcopy(hits),
            "trace": {
                "pageindex_selected_nodes": selected_nodes[:top_k],
                "tree_search_prompt_doc_count": len(candidate_base_ids),
            },
        }

    def base_status(self, base_id: str) -> dict[str, Any]:
        base_cache_dir = self.cache_dir / base_id
        cache_count = len(list(base_cache_dir.glob("*.json"))) if base_cache_dir.exists() else 0
        return {
            "base_id": base_id,
            "status": "ready",
            "pageindex_cache_count": cache_count,
        }
