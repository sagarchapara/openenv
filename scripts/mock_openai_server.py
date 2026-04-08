from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body or b"{}")
        except Exception:
            payload = {}

        messages = payload.get("messages", [])
        user_text = messages[-1].get("content", "") if messages else ""
        lower = user_text.lower()

        if "question:" in lower:
            if "what f1 score did the model with 16k token context achieve on scrolls" in lower:
                content = "43.2%"
            elif "by what percentage does sgc reduce memory consumption" in lower:
                content = "60%"
            elif "how many tokens was medllm trained on" in lower:
                content = "200 billion tokens"
            else:
                content = "42"
        else:
            content = "Concise summary preserving key facts and figures."

        response = {
            "id": "mock-chatcmpl",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
