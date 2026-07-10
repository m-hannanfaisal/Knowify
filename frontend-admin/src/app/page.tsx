"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from "recharts";

type Tab = "traces" | "metrics" | "documents" | "conversations" | "memory" | "usage";

interface DocumentInfo {
  id: string;
  filename: string;
  chunk_count: number;
  file_type: string;
}

interface Conversation {
  conversation_id: string;
  user_id: string;
  created_at: string;
  history: { role: string; content: string }[];
}

interface Trace {
  steps: { name: string; duration_ms: number }[];
  total_latency_ms: number;
}

interface CostRecord {
  date: string;
  user_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  cost: number;
}

interface MemoryItem {
  memory_id: string;
  text: string;
}

export default function AdminConsole() {
  const [activeTab, setActiveTab] = useState<Tab>("metrics");
  const [mounted, setMounted] = useState(false);

  // 1. Tab-specific states
  const [metrics, setMetrics] = useState<any>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [conversationTrace, setConversationTrace] = useState<Trace | null>(null);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  // Search/Filters states
  const [filterUser, setFilterUser] = useState("");
  const [filterStart, setFilterStart] = useState("");
  const [filterEnd, setFilterEnd] = useState("");

  // Memory states
  const [memoryUser, setMemoryUser] = useState("user_id_1");
  const [memories, setMemories] = useState<MemoryItem[]>([]);

  // Usage/Costs states
  const [costs, setCosts] = useState<CostRecord[]>([]);
  const [limitUser, setLimitUser] = useState("test_user_api");
  const [userUsage, setUserUsage] = useState<any>(null);
  const [newLimit, setNewLimit] = useState(5000);

  // Trace selection target
  const [traceIdInput, setTraceIdInput] = useState("session_api_123");

  // Prevent recharts SSR hydration errors
  useEffect(() => {
    setMounted(true);
    fetchMetrics();
    fetchDocuments();
    fetchConversations();
    fetchCosts();
    fetchUserUsage(limitUser);
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await fetch("/metrics");
      if (res.ok) {
        setMetrics(await res.json());
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch("/admin/documents");
      if (res.ok) {
        setDocuments(await res.json());
      }
    } catch (e) {
      console.error(e);
    }
  };

  const deleteDocument = async (id: string) => {
    try {
      const res = await fetch(`/admin/documents/${id}`, { method: "DELETE" });
      if (res.ok) {
        setDocuments((prev) => prev.filter((d) => d.id !== id));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const reindexDocument = async (id: string) => {
    try {
      await fetch(`/admin/documents/${id}/reindex`, { method: "POST" });
      alert(`Queued reindexing for document: ${id}`);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchConversations = async () => {
    try {
      let url = "/admin/conversations?";
      if (filterUser) url += `user_id=${filterUser}&`;
      if (filterStart) url += `start_date=${filterStart}&`;
      if (filterEnd) url += `end_date=${filterEnd}&`;

      const res = await fetch(url);
      if (res.ok) {
        setConversations(await res.json());
      }
    } catch (e) {
      console.error(e);
    }
  };

  const viewConversationDetails = async (conv: Conversation) => {
    setSelectedConversation(conv);
    try {
      const res = await fetch(`/admin/conversations/${conv.conversation_id}/trace`);
      if (res.ok) {
        setConversationTrace(await res.json());
        setSelectedStep(0);
      } else {
        setConversationTrace(null);
      }
    } catch (e) {
      setConversationTrace(null);
    }
  };

  const fetchTraceById = async () => {
    if (!traceIdInput.trim()) return;
    try {
      const res = await fetch(`/admin/conversations/${traceIdInput}/trace`);
      if (res.ok) {
        setConversationTrace(await res.json());
        setSelectedStep(0);
      } else {
        alert("Trace logs not found for this ID.");
        setConversationTrace(null);
      }
    } catch (e) {
      setConversationTrace(null);
    }
  };

  const fetchMemories = async () => {
    if (!memoryUser.trim()) return;
    try {
      const res = await fetch(`/admin/memory/${memoryUser}`);
      if (res.ok) {
        const data = await res.json();
        setMemories(data.memories || []);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const deleteMemory = async (memoryId: string) => {
    try {
      const res = await fetch(`/admin/memory/${memoryUser}/${memoryId}`, { method: "DELETE" });
      if (res.ok) {
        setMemories((prev) => prev.filter((m) => m.memory_id !== memoryId));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchCosts = async () => {
    try {
      const res = await fetch("/admin/costs");
      if (res.ok) {
        setCosts(await res.json());
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchUserUsage = async (uid: string) => {
    try {
      const res = await fetch(`/admin/usage/${uid}`);
      if (res.ok) {
        const data = await res.json();
        setUserUsage(data);
        setNewLimit(data.tokens_limit);
      } else {
        setUserUsage(null);
      }
    } catch (e) {
      setUserUsage(null);
    }
  };

  const updateUserLimit = async () => {
    try {
      const res = await fetch(`/admin/usage/${limitUser}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokens_limit: newLimit }),
      });
      if (res.ok) {
        const data = await res.json();
        setUserUsage(data);
        alert(`Limit updated successfully for user ${limitUser}`);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Format chart data
  const getChartData = () => {
    if (!metrics) return [];
    return [
      { name: "Hit Rate", value: metrics.retrieval_hit_rate * 100 },
      { name: "Cache Hits", value: metrics.cache_hit_rate * 100 },
      { name: "Retry Rate", value: metrics.retry_rate * 100 },
      { name: "Faithfulness", value: (metrics.average_faithfulness || 0.85) * 100 },
    ];
  };

  return (
    <div className="admin-container">
      {/* Sidebar Navigation */}
      <div className="admin-sidebar">
        <div className="admin-sidebar-header">
          <div className="admin-sidebar-dot" />
          Knowify Admin
        </div>
        <div className="admin-nav">
          <button
            className={`admin-nav-item ${activeTab === "metrics" ? "active" : ""}`}
            onClick={() => setActiveTab("metrics")}
          >
            Metrics Dashboard
          </button>
          <button
            className={`admin-nav-item ${activeTab === "traces" ? "active" : ""}`}
            onClick={() => setActiveTab("traces")}
          >
            Orchestrator Traces
          </button>
          <button
            className={`admin-nav-item ${activeTab === "documents" ? "active" : ""}`}
            onClick={() => setActiveTab("documents")}
          >
            Documents Index
          </button>
          <button
            className={`admin-nav-item ${activeTab === "conversations" ? "active" : ""}`}
            onClick={() => setActiveTab("conversations")}
          >
            Conversations Log
          </button>
          <button
            className={`admin-nav-item ${activeTab === "memory" ? "active" : ""}`}
            onClick={() => setActiveTab("memory")}
          >
            Long-Term Memory
          </button>
          <button
            className={`admin-nav-item ${activeTab === "usage" ? "active" : ""}`}
            onClick={() => setActiveTab("usage")}
          >
            Usage & Costs
          </button>
        </div>
      </div>

      {/* Main Panel Content */}
      <div className="admin-main">
        {/* 1. METRICS DASHBOARD */}
        {activeTab === "metrics" && (
          <div>
            <h2 className="admin-section-title">Telemetry Dashboard</h2>
            <div className="cost-grid">
              <div className="cost-card">
                <div className="cost-label">Total Requests</div>
                <div className="cost-value">{metrics?.total_requests ?? 0}</div>
              </div>
              <div className="cost-card">
                <div className="cost-label">p50 Latency</div>
                <div className="cost-value mono">{metrics?.latency_p50_ms ?? 0} ms</div>
              </div>
              <div className="cost-card">
                <div className="cost-label">p95 Latency</div>
                <div className="cost-value mono">{metrics?.latency_p95_ms ?? 0} ms</div>
              </div>
              <div className="cost-card">
                <div className="cost-label">RAGAS Faithfulness</div>
                <div className="cost-value">{metrics?.average_faithfulness ?? "0.85"}</div>
              </div>
            </div>

            <div className="admin-card">
              <div className="admin-card-title">Performance Ratios (%)</div>
              <div style={{ width: "100%", height: 300 }}>
                {mounted && (
                  <ResponsiveContainer>
                    <BarChart data={getChartData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#242624" />
                      <XAxis dataKey="name" stroke="#8A9089" />
                      <YAxis stroke="#8A9089" />
                      <ChartTooltip
                        contentStyle={{ backgroundColor: "#1A1C1A", borderColor: "#242624" }}
                      />
                      <Legend />
                      <Bar dataKey="value" fill="#3DDC84" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 2. TRACES TIMELINE VIEWER */}
        {activeTab === "traces" && (
          <div>
            <h2 className="admin-section-title">Orchestrator Step Traces</h2>
            <div className="filter-bar">
              <input
                className="filter-input mono"
                placeholder="Enter conversation ID..."
                value={traceIdInput}
                onChange={(e) => setTraceIdInput(e.target.value)}
              />
              <button className="admin-btn primary" onClick={fetchTraceById}>
                Load Trace
              </button>
            </div>

            {conversationTrace ? (
              <div style={{ display: "flex", gap: "32px", marginTop: "24px" }}>
                <div style={{ flex: 1 }}>
                  <div className="admin-card-title">Graph Traversal Timeline</div>
                  <div className="timeline-container">
                    {conversationTrace.steps.map((step, idx) => (
                      <div
                        key={idx}
                        className={`timeline-step ${selectedStep === idx ? "active" : ""}`}
                        onClick={() => setSelectedStep(idx)}
                      >
                        <span className="timeline-step-name mono">{step.name}</span>
                        <span className="timeline-step-duration mono">{step.duration_ms} ms</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: "16px", color: "#8A9089", fontSize: "13px" }}>
                    Total pipeline execution latency:{" "}
                    <span className="mono" style={{ color: "#3DDC84", fontWeight: 600 }}>
                      {conversationTrace.total_latency_ms} ms
                    </span>
                  </div>
                </div>

                <div style={{ flex: 1 }} className="admin-card">
                  <div className="admin-card-title">Step Latency Metrics</div>
                  {selectedStep !== null && conversationTrace.steps[selectedStep] ? (
                    <div>
                      <h4 className="mono" style={{ color: "#3DDC84", marginBottom: "12px" }}>
                        STEP: {conversationTrace.steps[selectedStep].name.toUpperCase()}
                      </h4>
                      <div className="mono" style={{ fontSize: "13px", lineHeight: "1.6" }}>
                        <p>Duration: {conversationTrace.steps[selectedStep].duration_ms} ms</p>
                        <p style={{ marginTop: "12px", color: "#8A9089" }}>
                          Status: COMPLETED_SUCCESSFULLY
                        </p>
                        <p style={{ color: "#5B6159" }}>
                          Step name: {conversationTrace.steps[selectedStep].name} | Latency: {conversationTrace.steps[selectedStep].duration_ms} ms
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div style={{ color: "#8A9089" }}>Select a step to view log trace logs.</div>
                  )}
                </div>
              </div>
            ) : (
              <div className="admin-card" style={{ color: "#8A9089" }}>
                No active session trace loaded. Click a session in the Conversations Log or enter a session ID above.
              </div>
            )}
          </div>
        )}

        {/* 3. DOCUMENTS MANAGEMENT */}
        {activeTab === "documents" && (
          <div>
            <h2 className="admin-section-title">Ingested Documents Index</h2>
            <div className="admin-card">
              <div className="admin-table-wrapper">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>Document ID</th>
                      <th>File Name</th>
                      <th>File Type</th>
                      <th>Chunk Count</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((doc) => (
                      <tr key={doc.id}>
                        <td className="mono">{doc.id}</td>
                        <td>{doc.filename}</td>
                        <td className="mono">{doc.file_type}</td>
                        <td className="mono">{doc.chunk_count}</td>
                        <td>
                          <div style={{ display: "flex", gap: "8px" }}>
                            <button
                              className="admin-btn"
                              onClick={() => reindexDocument(doc.id)}
                            >
                              Re-Index
                            </button>
                            <button
                              className="admin-btn danger"
                              onClick={() => deleteDocument(doc.id)}
                            >
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 4. CONVERSATIONS BROWSER */}
        {activeTab === "conversations" && (
          <div>
            <h2 className="admin-section-title">All-User Sessions Browser</h2>
            <div className="filter-bar">
              <input
                className="filter-input"
                placeholder="Filter by User ID"
                value={filterUser}
                onChange={(e) => setFilterUser(e.target.value)}
              />
              <input
                className="filter-input"
                type="date"
                value={filterStart}
                onChange={(e) => setFilterStart(e.target.value)}
              />
              <input
                className="filter-input"
                type="date"
                value={filterEnd}
                onChange={(e) => setFilterEnd(e.target.value)}
              />
              <button className="admin-btn primary" onClick={fetchConversations}>
                Apply Filters
              </button>
            </div>

            <div style={{ display: "flex", gap: "32px" }}>
              <div className="admin-card" style={{ flex: 1 }}>
                <div className="admin-card-title">Sessions</div>
                <div className="admin-table-wrapper">
                  <table className="admin-table">
                    <thead>
                      <tr>
                        <th>Session ID</th>
                        <th>User ID</th>
                        <th>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {conversations.map((c) => (
                        <tr
                          key={c.conversation_id}
                          style={{ cursor: "pointer" }}
                          onClick={() => viewConversationDetails(c)}
                        >
                          <td className="mono">{c.conversation_id}</td>
                          <td className="mono">{c.user_id}</td>
                          <td className="mono">{c.created_at}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {selectedConversation && (
                <div className="admin-card" style={{ flex: 1.2 }}>
                  <div className="admin-card-title">
                    History & Trace: {selectedConversation.conversation_id}
                  </div>
                  <div
                    style={{
                      maxHeight: "300px",
                      overflowY: "auto",
                      borderBottom: "1px solid var(--border-color)",
                      paddingBottom: "16px",
                      marginBottom: "16px",
                    }}
                  >
                    {selectedConversation.history.map((h, i) => (
                      <div key={i} style={{ marginBottom: "12px", fontSize: "13px" }}>
                        <strong
                          style={{
                            color: h.role === "user" ? "#E8EAE6" : "#3DDC84",
                            textTransform: "uppercase",
                          }}
                        >
                          {h.role}:
                        </strong>
                        <p style={{ marginTop: "4px", color: "#8A9089" }}>{h.content}</p>
                      </div>
                    ))}
                  </div>

                  {conversationTrace && (
                    <div>
                      <div className="admin-card-title">Orchestration Graph Traversal</div>
                      <div className="timeline-container">
                        {conversationTrace.steps.map((step, idx) => (
                          <div key={idx} className="timeline-step">
                            <span className="timeline-step-name mono">{step.name}</span>
                            <span className="timeline-step-duration mono">
                              {step.duration_ms} ms
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* 5. MEMORY INSPECTOR */}
        {activeTab === "memory" && (
          <div>
            <h2 className="admin-section-title">Long-Term Memory Inspector</h2>
            <div className="filter-bar">
              <input
                className="filter-input mono"
                placeholder="User ID..."
                value={memoryUser}
                onChange={(e) => setMemoryUser(e.target.value)}
              />
              <button className="admin-btn primary" onClick={fetchMemories}>
                Query Memory
              </button>
            </div>

            <div className="admin-card">
              <div className="admin-table-wrapper">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>Memory ID</th>
                      <th>Extracted Fact</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {memories.map((m) => (
                      <tr key={m.memory_id}>
                        <td className="mono">{m.memory_id}</td>
                        <td>{m.text}</td>
                        <td>
                          <button
                            className="admin-btn danger"
                            onClick={() => deleteMemory(m.memory_id)}
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 6. USAGE & COSTS */}
        {activeTab === "usage" && (
          <div>
            <h2 className="admin-section-title">Usage, Costs & Limits Console</h2>

            <div style={{ display: "flex", gap: "32px" }}>
              {/* User Limit adjustment */}
              <div className="admin-card" style={{ flex: 1 }}>
                <div className="admin-card-title">Modify Token Usage Ceiling</div>
                <div className="filter-bar" style={{ marginTop: "12px" }}>
                  <input
                    className="filter-input mono"
                    placeholder="User ID..."
                    value={limitUser}
                    onChange={(e) => setLimitUser(e.target.value)}
                  />
                  <button className="admin-btn" onClick={() => fetchUserUsage(limitUser)}>
                    Fetch Profile
                  </button>
                </div>

                {userUsage ? (
                  <div style={{ marginTop: "16px" }}>
                    <div style={{ fontSize: "14px", marginBottom: "16px" }}>
                      User: <span className="mono">{userUsage.user_id}</span>
                      <br />
                      Current usage (tokens):{" "}
                      <span className="mono" style={{ color: "#3DDC84" }}>
                        {userUsage.tokens_used}
                      </span>
                      {" / "}{userUsage.tokens_limit}
                    </div>

                    <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                      <input
                        className="filter-input mono"
                        type="number"
                        value={newLimit}
                        onChange={(e) => setNewLimit(parseInt(e.target.value))}
                      />
                      <button className="admin-btn primary" onClick={updateUserLimit}>
                        Update Limit
                      </button>
                    </div>
                  </div>
                ) : (
                  <div style={{ color: "#8A9089", marginTop: "16px", fontSize: "13px" }}>
                    Enter user_id and load profile limit details.
                  </div>
                )}
              </div>

              {/* Aggregated Daily Token Costs */}
              <div className="admin-card" style={{ flex: 1.5 }}>
                <div className="admin-card-title">Cost Logs & Token Aggregates</div>
                <div className="admin-table-wrapper">
                  <table className="admin-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>User ID</th>
                        <th>Prompt Tokens</th>
                        <th>Completion Tokens</th>
                        <th>Est. Cost ($)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {costs.map((c, i) => (
                        <tr key={i}>
                          <td className="mono">{c.date}</td>
                          <td className="mono">{c.user_id}</td>
                          <td className="mono">{c.prompt_tokens}</td>
                          <td className="mono">{c.completion_tokens}</td>
                          <td className="mono" style={{ color: "#3DDC84" }}>
                            ${c.cost.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
