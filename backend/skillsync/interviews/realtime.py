import asyncio
import base64
import json
from typing import Dict

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v2/interviews", tags=["AI Interviews - Realtime"])


class Blob:
    def __init__(self, data: bytes, mime_type: str):
        self.data = data
        self.mime_type = mime_type


class LiveQueue:
    def __init__(self) -> None:
        self._events: asyncio.Queue = asyncio.Queue()
        self._incoming: asyncio.Queue = asyncio.Queue()
        self._closed = False

    @property
    def events(self) -> asyncio.Queue:
        return self._events

    @property
    def incoming(self) -> asyncio.Queue:
        return self._incoming

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._events.put({"type": "control", "text": "Voice session closed"})


active_sessions: Dict[str, LiveQueue] = {}


def _session_key(user_id: str, interview_id: str | int) -> str:
    return f"{user_id}:{interview_id}"


async def start_agent_session(user_id: str, interview_id: str) -> LiveQueue:
    queue = LiveQueue()

    async def agent_task() -> None:
        try:
            while True:
                blob: Blob | None = await queue.incoming.get()
                if blob is None:
                    break
                await queue.events.put(
                    {
                        "type": "agent",
                        "mime_type": "text/plain",
                        "text": "Audio chunk received",
                        "bytes": len(blob.data),
                    }
                )
        except Exception as exc:  # pragma: no cover - diagnostic path
            await queue.events.put(
                {"type": "control", "text": f"Voice bridge error: {exc}"}
            )
        finally:
            await queue.events.put(
                {"type": "control", "text": "Voice bridge idle"}
            )

    asyncio.create_task(agent_task())
    return queue


async def agent_to_client_sse(queue: LiveQueue):
    try:
        while True:
            event = await queue.events.get()
            payload = f"data: {json.dumps(event)}\n\n"
            yield payload.encode("utf-8")
    except asyncio.CancelledError:  # pragma: no cover - cleanup path
        return


@router.get("/events/{user_id}/{interview_id}")
async def sse_endpoint(user_id: str, interview_id: str):
    key = _session_key(user_id, interview_id)
    queue = await start_agent_session(user_id, interview_id)
    active_sessions[key] = queue

    async def event_generator():
        try:
            await queue.events.put(
                {"type": "system", "text": "Voice session ready"}
            )
            async for chunk in agent_to_client_sse(queue):
                yield chunk
        finally:
            await queue.close()
            active_sessions.pop(key, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.post("/send/{user_id}")
async def send_message_endpoint(user_id: str, request: Request):
    payload = await request.json()
    interview_id = str(payload.get("interview_id"))
    key = _session_key(str(user_id), interview_id)
    queue = active_sessions.get(key)
    if not queue:
        return {"error": "Session not found"}

    mime_type = payload.get("mime_type")
    data = payload.get("data")

    if mime_type != "audio/pcm" or not data:
        return {"error": "Unsupported payload"}

    decoded = base64.b64decode(data)
    await queue.incoming.put(Blob(decoded, mime_type))
    return {"status": "accepted", "bytes": len(decoded)}


@router.websocket("/live/{interview_id}/ws")
async def websocket_bridge(websocket: WebSocket, interview_id: str):
    """Minimal WebSocket bridge so the frontend voice mode can connect."""
    user_id = "web"
    key = _session_key(user_id, interview_id)
    queue = await start_agent_session(user_id, interview_id)
    active_sessions[key] = queue

    await websocket.accept()
    await queue.events.put({"type": "system", "text": "Voice session connected"})

    async def agent_to_ws():
        try:
            while True:
                event = await queue.events.get()
                await websocket.send_text(json.dumps(event))
        except WebSocketDisconnect:
            pass

    async def ws_to_agent():
        try:
            while True:
                message = await websocket.receive()
                data_bytes = message.get("bytes")
                data_text = message.get("text")
                if data_bytes:
                    await queue.incoming.put(Blob(data_bytes, "audio/webm"))
                elif data_text:
                    await queue.events.put({"type": "client", "text": data_text})
        except WebSocketDisconnect:
            pass

    sender = asyncio.create_task(agent_to_ws())
    receiver = asyncio.create_task(ws_to_agent())

    try:
        await asyncio.gather(sender, receiver)
    finally:
        sender.cancel()
        receiver.cancel()
        await queue.close()
        active_sessions.pop(key, None)
