def test_knowledge_routes_and_debug_queries(client):
    create_resp = client.post(
        "/api/v1/knowledge/bases",
        json={
            "schema_version": "1.0",
            "meta": {"source": "test"},
            "payload": {
                "name": "心理手册",
                "route": "hybrid",
                "knowledge_type": "manual",
                "namespace": "default",
                "tags": ["manual"],
            },
            "extensions": {},
        },
    )
    assert create_resp.status_code == 200
    base_id = create_resp.json()["payload"]["base_id"]

    ingest_resp = client.post(
        f"/api/v1/knowledge/bases/{base_id}/documents",
        json={
            "schema_version": "1.0",
            "meta": {"source": "test"},
            "payload": {
                "documents": [
                    {
                        "title": "老年焦虑干预手册",
                        "content": "# 睡眠卫生\n帮助老人建立规律作息，缓解压力。",
                        "metadata": {"audience": "elder"},
                    }
                ]
            },
            "extensions": {},
        },
    )
    assert ingest_resp.status_code == 200

    query_resp = client.post(
        "/api/v1/knowledge/query",
        json={
            "schema_version": "1.0",
            "meta": {"source": "test"},
            "payload": {
                "query": "睡眠 压力",
                "options": {"mode": "auto", "knowledge_type": "manual", "document_scope": "long_doc", "filters": {"base_id": base_id}},
                "debug": {"trace": True},
            },
            "extensions": {},
        },
    )
    assert query_resp.status_code == 200
    assert query_resp.json()["payload"]["retrieval_context"]["pageindex_hits"]

    debug_resp = client.post(
        "/api/v1/knowledge/query/milvus",
        json={
            "schema_version": "1.0",
            "meta": {"source": "test"},
            "payload": {
                "query": "睡眠 压力",
                "options": {"mode": "milvus", "filters": {"base_id": base_id}},
                "debug": {"trace": False},
            },
            "extensions": {},
        },
    )
    assert debug_resp.status_code == 200
    assert debug_resp.json()["payload"]["route_used"] == "milvus"
