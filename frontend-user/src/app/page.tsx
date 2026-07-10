"use client";

import { useEffect, useRef, useState } from "react";

// Helper to generate simple unique session IDs
const generateSessionId = () => {
  return "session_" + Math.random().toString(36).substring(2, 11);
};

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: any[];
}

interface FileUploadStatus {
  name: string;
  status: "loading" | "success" | "error";
  chunks?: number;
}

export default function Home() {
  const [userId] = useState("default_user");
  const [conversationId, setConversationId] = useState("");
  const [sessions, setSessions] = useState<string[]>([]);
  const [sessionTitles, setSessionTitles] = useState<Record<string, string>>({});
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputVal, setInputVal] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [usage, setUsage] = useState({ tokens_used: 0, tokens_limit: 5000, resets_at: "" });
  const [uploads, setUploads] = useState<FileUploadStatus[]>([]);
  const [activePopover, setActivePopover] = useState<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize session
  useEffect(() => {
    const defaultSession = generateSessionId();
    setConversationId(defaultSession);
    setSessions([defaultSession]);
    fetchUsageLimit();
  }, []);

  // Scroll chat area to bottom when messages are appended
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const updateSessionTitle = (sid: string, titleText: string) => {
    setSessionTitles((prev) => ({
      ...prev,
      [sid]: titleText.substring(0, 24) + (titleText.length > 24 ? "..." : ""),
    }));
  };

  const fetchAllSessionTitles = async (sids: string[]) => {
    for (const sid of sids) {
      try {
        const res = await fetch(`/api/v1/conversations/${sid}`);
        if (res.ok) {
          const data = await res.json();
          const firstUserMsg = data.history?.find((m: any) => m.role === "user")?.content;
          if (firstUserMsg) {
            updateSessionTitle(sid, firstUserMsg);
          }
        }
      } catch (e) {
        console.error(e);
      }
    }
  };

  const fetchUsageLimit = async () => {
    try {
      const res = await fetch(`/api/v1/usage/${userId}`);
      if (res.ok) {
        const data = await res.json();
        setUsage({
          tokens_used: data.tokens_used,
          tokens_limit: data.tokens_limit,
          resets_at: data.resets_at || "",
        });
      }
    } catch (e) {
      console.error("Failed to fetch limits", e);
    }
  };


  const startNewChat = () => {
    const newSession = generateSessionId();
    setConversationId(newSession);
    setSessions((prev) => [newSession, ...prev]);
    setMessages([]);
    setUploads([]);
  };

  const switchSession = async (sid: string) => {
    setConversationId(sid);
    setMessages([]);
    setUploads([]);
    try {
      const res = await fetch(`/api/v1/conversations/${sid}`);
      if (res.ok) {
        const data = await res.json();
        const history = data.history || [];
        setMessages(history);
        const firstUserMsg = history.find((m: any) => m.role === "user")?.content;
        if (firstUserMsg) {
          updateSessionTitle(sid, firstUserMsg);
        }
      }
    } catch (e) {
      console.error("Failed to fetch history", e);
    }
  };

  const handleFileUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    const newUpload: FileUploadStatus = { name: file.name, status: "loading" };
    setUploads((prev) => [newUpload, ...prev]);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("collection_name", "knowify_collection");

    try {
      const res = await fetch("/api/v1/upload", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploads((prev) =>
          prev.map((u) =>
            u.name === file.name
              ? { name: file.name, status: "success", chunks: data.chunks_processed }
              : u
          )
        );
      } else {
        throw new Error();
      }
    } catch (err) {
      setUploads((prev) =>
        prev.map((u) => (u.name === file.name ? { name: file.name, status: "error" } : u))
      );
    }
  };

  const handleSend = async (text: string) => {
    if (!text.trim() || isStreaming) return;

    if (messages.length === 0) {
      updateSessionTitle(conversationId, text);
    }

    setInputVal("");
    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);

    const assistantMsg: Message = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          user_id: userId,
          conversation_id: conversationId,
        }),
      });

      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = "";

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

        const packets = buffer.split("\n\n");
        buffer = packets.pop() || "";

        for (const packet of packets) {
          if (packet.startsWith("data: ")) {
            try {
              const payload = JSON.parse(packet.substring(6));
              if (payload.type === "token") {
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.role === "assistant") {
                    last.content += payload.text;
                  }
                  return updated;
                });
              } else if (payload.type === "citations") {
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.role === "assistant") {
                    last.citations = payload.citations;
                  }
                  return updated;
                });
              } else if (payload.type === "metrics") {
                const tokensUsed = payload.token_counts?.total_tokens || 0;
                setUsage((prev) => ({
                  ...prev,
                  tokens_used: prev.tokens_used + tokensUsed,
                }));
              }
            } catch (e) {
              // Fragmented JSON packet
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.role === "assistant") {
          last.content = "An error occurred while communicating with the backend.";
        }
        return updated;
      });
    } finally {
      setIsStreaming(false);
      fetchUsageLimit();
    }
  };

  // Inline citations chip popover renderer
  const renderMessageContent = (msg: Message, messageIndex: number) => {
    const content = msg.content;
    const citations = msg.citations || [];
    const parts = [];
    const regex = /\[source:\s*([^,\]]+),\s*([^\]]+)\]/g;
    let lastIndex = 0;
    let match;
    let chipIndex = 1;

    while ((match = regex.exec(content)) !== null) {
      const filename = match[1].trim();
      const location = match[2].trim();

      if (match.index > lastIndex) {
        parts.push(content.substring(lastIndex, match.index));
      }

      const currentChipIndex = chipIndex++;
      const popoverKey = `${messageIndex}-${match.index}`;

      parts.push(
        <div key={popoverKey} className="citation-chip-container">
          <span
            className="citation-chip"
            onClick={(e) => {
              e.stopPropagation();
              setActivePopover(activePopover === popoverKey ? null : popoverKey);
            }}
          >
            {currentChipIndex}
          </span>
          {activePopover === popoverKey && (
            <div className="citation-popover">
              <div className="citation-popover-title">{filename}</div>
              <div className="citation-popover-location">{location}</div>
            </div>
          )}
        </div>
      );

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return parts.length > 0 ? parts : content;
  };

  const formatResetsAt = (isoString: string) => {
    if (!isoString) return "resets at 24h";
    try {
      const date = new Date(isoString);
      if (isNaN(date.getTime())) return "resets at 24h";
      const now = new Date();
      const diffMs = date.getTime() - now.getTime();
      if (diffMs > 0) {
        const diffHours = Math.ceil(diffMs / (1000 * 60 * 60));
        return `resets in ${diffHours}h`;
      }
      return "resets at " + date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    } catch (e) {
      return "resets at 24h";
    }
  };

  const remaining = Math.max(0, usage.tokens_limit - usage.tokens_used);
  const isLimitReached = usage.tokens_used >= usage.tokens_limit;


  return (
    <div className="app-container" onClick={() => setActivePopover(null)}>
      {/* 1. Left Sidebar: Scoped Session History */}
      <div className="sidebar">
        <div className="logo-container">
          <div className="logo-dot" />
          <div className="logo-text">Knowify</div>
        </div>

        <button className="new-chat-btn" onClick={startNewChat}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          New chat
        </button>

        <div className="history-list">
          {sessions.map((sid) => (
            <div
              key={sid}
              className={`history-item ${conversationId === sid ? "active" : ""}`}
              onClick={() => switchSession(sid)}
            >
              {sessionTitles[sid] || sid}
            </div>
          ))}
        </div>
      </div>

      {/* 2. Main Right Column */}
      <div className="main-content">
        <div className="header">
          <div className="user-tag">Logged in as {userId}</div>
          <div className="usage-pill">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
              <path d="M11 21h-1l1-7H5l9-11h1l-1 7h6z" />
            </svg>
            {remaining} tokens remaining
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-area">
          {messages.length === 0 ? (
            /* Empty conversation state with AcmeCRM suggested prompt cards */
            <div className="empty-state">
              <div className="greeting-text">How can I help you today?</div>
              <div className="prompt-suggestions">
                <div
                  className="prompt-card"
                  onClick={() => handleSend("How do I create an API key?")}
                >
                  <h4>API Key Creation</h4>
                  <p>How do I create an API key?</p>
                </div>
                <div
                  className="prompt-card"
                  onClick={() => handleSend("What's included in the Professional plan?")}
                >
                  <h4>Professional Plan</h4>
                  <p>What's included in the Professional plan?</p>
                </div>
                <div
                  className="prompt-card"
                  onClick={() => handleSend("Why is webhook verification failing?")}
                >
                  <h4>Webhook Failures</h4>
                  <p>Why is webhook verification failing?</p>
                </div>
                <div
                  className="prompt-card"
                  onClick={() => handleSend("Can I export my contacts?")}
                >
                  <h4>Exporting Contacts</h4>
                  <p>Can I export my contacts?</p>
                </div>
              </div>

              {/* Upload Widget */}
              <div className="upload-container" onClick={handleFileUploadClick}>
                <div className="upload-text">
                  Add your first document to get started. <span>Click to upload</span> (PDF/DOCX/HTML/CSV/XLSX/JSON/Images)
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: "none" }}
                  onChange={handleFileChange}
                  accept=".pdf,.docx,.html,.csv,.xlsx,.json,.jpg,.jpeg,.png"
                />

                {uploads.length > 0 && (
                  <div className="file-status-list">
                    {uploads.map((u, i) => (
                      <div key={i} className="file-status-item">
                        <div className="file-status-info">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                          </svg>
                          <span>{u.name}</span>
                        </div>
                        <span className={`file-status-badge ${u.status === "success" ? "success" : "loading"}`}>
                          {u.status === "loading"
                            ? "Ingesting..."
                            : u.status === "success"
                            ? `Ingested (${u.chunks} chunks)`
                            : "Error"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Dialogue feeds */
            messages.map((msg, index) => (
              <div key={index} className={`message-bubble ${msg.role}`}>
                <div className="message-sender">{msg.role === "user" ? "You" : "Knowify"}</div>
                <div className="message-text">{renderMessageContent(msg, index)}</div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-area">
          <div className="input-pill-wrapper">
            <input
              className="input-pill"
              placeholder={isLimitReached ? `Token limit reached — ${formatResetsAt(usage.resets_at)}` : "Ask anything..."}
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend(inputVal)}
              disabled={isStreaming || isLimitReached}
            />
            <button className="send-btn" onClick={() => handleSend(inputVal)} disabled={isStreaming || isLimitReached}>
              <svg width="18" height="18" viewBox="0 0 24 24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

  );
}
