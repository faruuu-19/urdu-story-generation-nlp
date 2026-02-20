import { NextRequest } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch(`${BACKEND_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorText = await response.text()
      return new Response(
        JSON.stringify({ error: `Backend error: ${response.status} - ${errorText}` }),
        { status: response.status, headers: { "Content-Type": "application/json" } }
      )
    }

    // Stream the SSE response back to the client
    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    })
  } catch (error) {
    console.error("Proxy error:", error)
    return new Response(
      JSON.stringify({ error: "Failed to connect to story generation backend. Make sure the Python server is running on port 8000." }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    )
  }
}

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`)
    const data = await response.json()
    return Response.json(data)
  } catch {
    return Response.json(
      { status: "error", message: "Backend not reachable" },
      { status: 502 }
    )
  }
}
