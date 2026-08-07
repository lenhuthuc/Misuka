async def test_vad_predict_returns_scores(client):
    resp = await client.post("/vad", json={"text": "toi rat vui"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"v", "a", "d"}
