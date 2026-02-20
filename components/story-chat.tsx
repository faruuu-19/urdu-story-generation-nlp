"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { BookOpen, Send, Sparkles, AlertCircle, Loader2, Settings2, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface Message {
  id: string
  role: "user" | "assistant" | "error"
  content: string
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

export function StoryChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [maxLength, setMaxLength] = useState(200)
  const [showSettings, setShowSettings] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const generateStory = async () => {
    if (!input.trim() || isGenerating) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    }

    setMessages((prev) => [...prev, userMessage])
    const prefix = input.trim()
    setInput("")
    setIsGenerating(true)

    const assistantId = (Date.now() + 1).toString()
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "" },
    ])

    try {
      const response = await fetch(`${BACKEND_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prefix, max_length: maxLength }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response stream")

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim()
            if (!dataStr || dataStr === "{}") continue

            try {
              const data = JSON.parse(dataStr)
              if (data.token) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantId
                      ? { ...msg, content: msg.content + data.token }
                      : msg
                  )
                )
              }
              if (data.error) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantId
                      ? { ...msg, role: "error", content: data.error }
                      : msg
                  )
                )
              }
            } catch {
              // skip unparseable lines
            }
          }

          if (line.startsWith("event: done")) {
            break
          }
        }
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error"

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                role: "error",
                content: `کہانی بنانے میں خرابی ہوئی: ${errorMessage}. براہ کرم یقینی بنائیں کہ Python بیک اینڈ سرور چل رہا ہے۔`,
              }
            : msg
        )
      )
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="flex flex-col h-dvh bg-background">
      {/* Header */}
      <header className="flex-shrink-0 flex items-center justify-center gap-3 px-4 py-5 bg-primary text-primary-foreground">
        <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-primary-foreground/15">
          <BookOpen className="w-6 h-6" />
        </div>
        <div className="text-center">
          <h1 className="text-2xl font-bold leading-tight">
            {" ریختہ "}
          </h1>
          <p className="text-sm opacity-80">
            {"اردو میں بچوں کے لیے جادوئی کہانیاں"}
          </p>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6" dir="rtl">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-muted-foreground">
            <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
              <Sparkles className="w-10 h-10 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-lg font-medium text-foreground">
                {"!خوش آمدید"}
              </p>
              <p className="text-sm mt-1">
                {"کہانی کا آغاز لکھیں اور ہم اسے مکمل کریں گے"}
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 mt-2 max-w-md">
              {["ایک دفعہ کا ذکر ہے", "ایک شہزادی تھی", "جنگل میں ایک"].map(
                (suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="px-4 py-2 rounded-lg bg-secondary text-secondary-foreground text-sm hover:bg-secondary/80 transition-colors cursor-pointer"
                  >
                    {suggestion}
                  </button>
                )
              )}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-4 max-w-2xl mx-auto">
          {messages.map((message) => (
            <ChatBubble key={message.id} message={message} />
          ))}

          {isGenerating && (
            <div className="flex items-center gap-2 text-muted-foreground text-sm pr-12">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>{"...کہانی لکھی جا رہی ہے"}</span>
            </div>
          )}
        </div>
        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <footer className="flex-shrink-0 border-t border-border bg-card px-4 py-3">
        <div className="max-w-2xl mx-auto">
          {showSettings && (
            <div className="flex items-center gap-3 mb-3 p-3 bg-secondary rounded-lg" dir="rtl">
              <label className="text-sm text-secondary-foreground whitespace-nowrap">
                {"زیادہ سے زیادہ الفاظ:"}
              </label>
              <input
                type="range"
                min={50}
                max={500}
                step={50}
                value={maxLength}
                onChange={(e) => setMaxLength(Number(e.target.value))}
                className="flex-1 accent-primary"
              />
              <span className="text-sm font-medium text-secondary-foreground min-w-[3ch] text-center">
                {maxLength}
              </span>
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault()
              generateStory()
            }}
            className="flex items-center gap-2"
            dir="rtl"
          >
            <Button
              type="submit"
              disabled={!input.trim() || isGenerating}
              className="rounded-xl px-5 h-11 bg-primary text-primary-foreground hover:bg-accent"
            >
              {isGenerating ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  <Send className="w-4 h-4 ml-2 rotate-180" />
                  {"کہانی بنائیں"}
                </>
              )}
            </Button>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="یہاں کہانی کا آغاز لکھیں..."
              disabled={isGenerating}
              className="flex-1 h-11 px-4 rounded-xl border border-input bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring text-right"
              dir="rtl"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setShowSettings(!showSettings)}
              className={cn(
                "rounded-xl h-11 w-11 shrink-0",
                showSettings && "bg-secondary"
              )}
            >
              <Settings2 className="w-5 h-5" />
              <span className="sr-only">Settings</span>
            </Button>
          </form>
        </div>
      </footer>
    </div>
  )
}

function ChatBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"
  const isError = message.role === "error"

  return (
    <div
      className={cn(
        "flex items-start gap-2",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center mt-1",
          isUser
            ? "bg-primary text-primary-foreground"
            : isError
              ? "bg-destructive/15 text-destructive"
              : "bg-primary/15 text-primary"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4" />
        ) : isError ? (
          <AlertCircle className="w-4 h-4" />
        ) : (
          <BookOpen className="w-4 h-4" />
        )}
      </div>

      {/* Bubble */}
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-sm"
            : isError
              ? "bg-destructive/10 text-destructive border border-destructive/20 rounded-tl-sm"
              : "bg-card text-card-foreground border border-border shadow-sm rounded-tl-sm"
        )}
        dir="rtl"
      >
        <p className="whitespace-pre-wrap text-right">{message.content}</p>
      </div>
    </div>
  )
}
