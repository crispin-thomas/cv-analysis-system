import type React from "react";
import { useEffect, useRef, useState } from "react";
import { Button, Input } from "rsuite";
import { Send } from "lucide-react";
import { queryCV } from "../api/postQueryCVs";
import type { Message } from "../types";

type Props = {
  height: number;
};

export function Chat({ height }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "system",
      content: "Welcome to the CV Query Assistant. How can I help you today?",
    },
  ]);

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await queryCV({
        query: input,
        conversation_id: "1",
        user_id: "1",
      });

      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "system",
          content: response.response,
        },
      ]);
    } catch (error) {
      console.error("Error processing query:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "system",
          content: "Sorry, I encountered an error processing your query.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container" style={{ height: height - 190 + "px" }}>
      <div className="messages-area">
        <div className="message-container">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${
                message.role === "user" ? "user-message" : "system-message"
              }`}
            >
              <p>{message.content}</p>
            </div>
          ))}
          {isLoading && (
            <div className="message system-message">
              <p>Thinking...</p>
            </div>
          )}
        </div>
      </div>
      <form onSubmit={handleSubmit} className="input-area">
        <Input
          placeholder="Ask about candidates, skills, education, or experience..."
          value={input}
          onChange={(e) => setInput(e)}
          disabled={isLoading}
          className="input-field"
        />
        <Button type="submit" disabled={isLoading} className="send-button" appearance="primary">
          <Send className="send-icon" />
        </Button>
      </form>
    </div>
  );
}
