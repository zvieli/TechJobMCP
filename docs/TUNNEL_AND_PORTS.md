# Cloudflare Tunnel & Port Export Guide

To connect external cloud AI platforms (such as **Gemini Spark**, **ChatGPT / Custom GPTs**, or **remote Claude agents**) to your locally running TechJobMCP server, you need to expose your local port (`8000`) over a secure public HTTPS tunnel.

We recommend **Cloudflare Tunnel (`cloudflared`)** for fast, zero-config, free HTTPS tunnels without opening router ports.

---

## 1. Download & Install `cloudflared`

Run the following commands in your terminal:

```bash
# 1. Download the latest Linux x86_64 cloudflared binary
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# 2. Make it executable
chmod +x cloudflared
```

*(For macOS: `brew install cloudflared`; for Windows: download `cloudflared-windows-amd64.exe`)*.

---

## 2. Start the HTTPS Tunnel

Make sure your TechJobMCP server is already running locally on port 8000 (`docker compose up -d` or `uv run python -m job_mcp`).

Start the tunnel:
```bash
./cloudflared tunnel --url http://localhost:8000
```

`cloudflared` will generate a public HTTPS URL looking like:
```text
https://xxxxxxxxxxxxxx.trycloudflare.com
```

### Running in the Background / Persistent Terminal
To keep the tunnel alive while working:

**Using `nohup` / background:**
```bash
nohup ./cloudflared tunnel --url http://localhost:8000 > cloudflared.log 2>&1 &
# View generated URL
grep -o 'https://.*\.trycloudflare\.com' cloudflared.log | head -n 1
```

**Using `tmux` or `screen`:**
```bash
tmux new -s tunnel
./cloudflared tunnel --url http://localhost:8000
# Press Ctrl+B then D to detach
```

---

## 3. Verifying the Public Tunnel

Test your tunnel URL with `curl`:

```bash
# Replace with your actual trycloudflare domain:
TUNNEL_URL="https://your-tunnel-id.trycloudflare.com"

# 1. Check Health
curl -s "$TUNNEL_URL/health"
# Expected output: {"status": "ok", "service": "TechJobMCP"}

# 2. Check FastMCP Endpoint
curl -I "$TUNNEL_URL/mcp"
# Expected output: HTTP/2 200 OK (or HTTP/1.1 200 OK)
```

---

## 4. Troubleshooting Tunnel Connectivity

- **`read: connection reset by peer` / `502 Bad Gateway`**:
  Ensure the TechJobMCP server is actually running on `http://localhost:8000`. Check `docker compose ps` or test `curl http://localhost:8000/health` locally.
- **`Unsupported transport protocol` error**:
  Ensure `MCP_TRANSPORT=http` in `docker-compose.yml` or `.env`. FastMCP runs an internal HTTP listener inside the container; public HTTPS termination is handled by Cloudflare.

Next, see [**AI Client Integrations Guide**](./CLIENT_INTEGRATIONS.md) to connect this URL to Gemini Spark, Claude, and ChatGPT.
