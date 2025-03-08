// import { cn } from "@/lib/utils"
import type { Message } from "../types"
import { User, Bot } from "lucide-react"

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  return (
    <div
      // className={cn("flex items-start gap-3 rounded-lg p-3", message.role === "user" ? "bg-muted/50" : "bg-background")}
    >
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border bg-background shadow">
        {message.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className="flex-1 space-y-2">
        <p className="text-sm leading-relaxed text-foreground">{message.content}</p>
      </div>
    </div>
  )
}

