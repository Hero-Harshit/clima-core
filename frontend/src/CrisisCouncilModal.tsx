import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BadgeCheck,
  Check,
  Clock,
  FileText,
  Flame,
  Radio,
  RefreshCw,
  Scale,
  ShieldCheck,
  Truck,
  Users,
  X,
  Zap,
} from "lucide-react";

type CouncilProposal = {
  member_id: string;
  title: string;
  core_priority: string;
  demands: string[];
  objections_to_others: Record<string, string>;
  statutory_standard: string;
  resource_request: string;
};

type DebateTurn = {
  turn_number: number;
  speaker_id: string;
  speaker_title: string;
  stance: string;
  argument: string;
};

type ExecutiveDirective = {
  directive_id: string;
  title: string;
  mandate: string;
  enforcement_window: string;
  affected_sectors: string[];
  mitigated_risk: string;
};

type CrisisCouncilResponse = {
  incident_id: string;
  city: string;
  incident_type: string;
  severity_tier: "CRITICAL" | "HIGH" | "MODERATE" | string;
  initiating_metrics: Record<string, unknown>;
  member_proposals: CouncilProposal[];
  debate_transcript: DebateTurn[];
  friction_matrix: Record<string, string[]>;
  executive_directives: ExecutiveDirective[];
  accepted_tradeoffs: string[];
  life_preservation_index: number;
  economic_continuity_index: number;
  verification_hash: string;
  ics_form_reference?: string;
  timestamp_ist?: string;
  latency_ms?: number;
};

type CrisisCouncilModalProps = {
  city: string;
  initialTempMax?: number;
  initialRainfall?: number;
  initialGridLoad?: number;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

export default function CrisisCouncilModal({
  city,
  initialTempMax,
  initialRainfall,
  initialGridLoad,
  onClose,
}: CrisisCouncilModalProps) {
  const [activeTab, setActiveTab] = useState<"directives" | "transcript" | "friction" | "proposals">("directives");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [councilData, setCouncilData] = useState<CrisisCouncilResponse | null>(null);

  // Scenario parameter controls
  const [tempMax, setTempMax] = useState<number | undefined>(initialTempMax);
  const [rainfall, setRainfall] = useState<number | undefined>(initialRainfall);
  const [gridLoad, setGridLoad] = useState<number | undefined>(initialGridLoad);

  const conveneCouncil = async (overrideTemp?: number, overrideRain?: number, overrideGrid?: number) => {
    setLoading(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = {
        city,
      };
      const t = overrideTemp !== undefined ? overrideTemp : tempMax;
      const r = overrideRain !== undefined ? overrideRain : rainfall;
      const g = overrideGrid !== undefined ? overrideGrid : gridLoad;

      if (t !== undefined) payload.temp_max_c = Number(t);
      if (r !== undefined) payload.rainfall_mm = Number(r);
      if (g !== undefined) payload.peak_grid_load_mw = Number(g);

      const res = await fetch(`${API_BASE}/api/crisis-council/deliberate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Crisis council deliberation failed (${res.status})`);
      }
      const data = (await res.json()) as CrisisCouncilResponse;
      setCouncilData(data);
    } catch (err) {
      console.error("Council error:", err);
      setError(err instanceof Error ? err.message : "Failed to convene crisis council.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void conveneCouncil();
  }, [city]);

  const severityClass = (councilData?.severity_tier ?? "HIGH").toLowerCase();

  const getMemberIcon = (memberId: string) => {
    if (memberId.includes("health")) return Activity;
    if (memberId.includes("grid")) return Zap;
    if (memberId.includes("coldchain") || memberId.includes("logistics")) return Truck;
    return Users;
  };

  const getMemberColor = (memberId: string) => {
    if (memberId.includes("health")) return "health";
    if (memberId.includes("grid")) return "grid";
    if (memberId.includes("coldchain") || memberId.includes("logistics")) return "logistics";
    return "general";
  };

  return (
    <div className="crisis-council-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <section className="crisis-council-modal" role="dialog" aria-modal="true" aria-labelledby="war-room-title">
        {/* Top Header */}
        <header className="war-room-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn war-room-circle-btn" style={{ width: 36, height: 36 }}>
              <span style={{ fontWeight: 800, fontSize: 16, lineHeight: 1 }}>!</span>
            </div>
            <div className="war-room-title-block">
              <div className="war-room-kicker-row">
                <span className="war-room-badge"><Radio size={12} className="live-ping-icon" /> ICS-201 INCIDENT COMMAND</span>
                <span className={`severity-tag severity-${severityClass}`}>
                  <span className="dot" />
                  {councilData?.severity_tier ?? "ACTIVE DELIBERATION"}
                </span>
              </div>
              <h2 id="war-room-title">Crisis Council War Room · <span>{city}</span></h2>
              <p className="war-room-incident-meta">
                Incident ID: <code>{councilData?.incident_id ?? "SYNTHESIZING..."}</code> · {councilData?.incident_type?.replace(/_/g, " ") ?? "MULTI-SECTOR SHOCK"}
              </p>
            </div>
          </div>
          <button className="icon-btn close-warroom-btn" onClick={onClose} aria-label="Exit War Room">
            <X size={18} />
          </button>
        </header>

        {/* Quick Scenario Preset Chips */}
        <div className="war-room-presets-bar">
          <span className="presets-label"><Flame size={13} /> STRESS PRESETS:</span>
          <button
            type="button"
            className="preset-chip"
            onClick={() => {
              setTempMax(46.5);
              setRainfall(0);
              setGridLoad(10800);
              void conveneCouncil(46.5, 0, 10800);
            }}
          >
            Extreme Heat Dome (46.5°C)
          </button>
          <button
            type="button"
            className="preset-chip"
            onClick={() => {
              setTempMax(28.0);
              setRainfall(220.0);
              void conveneCouncil(28.0, 220.0, gridLoad);
            }}
          >
            Pluvial Cloudburst (220mm)
          </button>
          <button
            type="button"
            className="preset-chip reset"
            onClick={() => {
              setTempMax(initialTempMax);
              setRainfall(initialRainfall);
              setGridLoad(initialGridLoad);
              void conveneCouncil(initialTempMax, initialRainfall, initialGridLoad);
            }}
          >
            <RefreshCw size={12} /> Live Baseline
          </button>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="war-room-loading">
            <div className="radar-spinner" />
            <h3>Convening Crisis Council</h3>
            <p>Deliberating positions across NDMA Public Health, SLDC Grid Operations &amp; Cold-Chain Logistics…</p>
          </div>
        )}

        {/* Error State */}
        {!loading && error && (
          <div className="war-room-error">
            <AlertTriangle size={24} />
            <p>{error}</p>
            <button type="button" className="retry-btn" onClick={() => void conveneCouncil()}>
              <RefreshCw size={14} /> Retry Deliberation
            </button>
          </div>
        )}

        {/* Content Body */}
        {!loading && councilData && (
          <div className="war-room-content">
            {/* Command KPI Bar */}
            <div className="war-room-kpis">
              <div className="kpi-card life-kpi">
                <div className="kpi-icon-wrap"><ShieldCheck size={20} /></div>
                <div>
                  <div className="kpi-label">LIFE PRESERVATION INDEX</div>
                  <div className="kpi-value">{councilData.life_preservation_index.toFixed(1)}<span>%</span></div>
                  <div className="kpi-sub">Priority: Zero preventable mortalities</div>
                </div>
                <div className="kpi-gauge">
                  <div className="kpi-gauge-fill" style={{ width: `${councilData.life_preservation_index}%` }} />
                </div>
              </div>

              <div className="kpi-card econ-kpi">
                <div className="kpi-icon-wrap"><Activity size={20} /></div>
                <div>
                  <div className="kpi-label">ECONOMIC CONTINUITY INDEX</div>
                  <div className="kpi-value">{councilData.economic_continuity_index.toFixed(1)}<span>%</span></div>
                  <div className="kpi-sub">Priority: Critical infrastructure uptime</div>
                </div>
                <div className="kpi-gauge">
                  <div className="kpi-gauge-fill" style={{ width: `${councilData.economic_continuity_index}%` }} />
                </div>
              </div>

              <div className="kpi-card shock-kpi">
                <div className="kpi-icon-wrap"><Zap size={20} /></div>
                <div>
                  <div className="kpi-label">INITIATING ENVIRONMENTAL SHOCK</div>
                  <div className="shock-metrics-row">
                    <span>Temp: <b>{String(councilData.initiating_metrics.temp_max_c ?? "--")}°C</b></span>
                    <span>Rain: <b>{String(councilData.initiating_metrics.rainfall_mm ?? "--")} mm</b></span>
                    <span>Grid: <b>{String(councilData.initiating_metrics.peak_grid_load_mw ?? "--")} MW</b></span>
                  </div>
                  <div className="kpi-sub">Statutory threshold breach detected</div>
                </div>
              </div>
            </div>

            {/* Sector Authority Position Papers */}
            <div className="war-room-section">
              <div className="section-title-row">
                <h3><Users size={15} /> CONVENED SECTOR AUTHORITIES</h3>
                <span className="sub-tag">3 CONCURRENT AGENTS</span>
              </div>
              <div className="authorities-grid">
                {councilData.member_proposals.map((member) => {
                  const Icon = getMemberIcon(member.member_id);
                  const colorClass = getMemberColor(member.member_id);
                  return (
                    <article key={member.member_id} className={`authority-card auth-${colorClass}`}>
                      <div className="authority-head">
                        <span className="auth-avatar"><Icon size={16} /></span>
                        <div>
                          <h4>{member.title}</h4>
                          <span className="auth-core-goal">{member.core_priority}</span>
                        </div>
                      </div>

                      <div className="auth-demands-list">
                        <span className="demands-kicker">KEY DEMANDS:</span>
                        <ul>
                          {member.demands.slice(0, 3).map((demand, i) => (
                            <li key={i}><Check size={12} className="check-icon" /> {demand}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="auth-statute-row">
                        <BadgeCheck size={13} />
                        <span>{member.statutory_standard}</span>
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>

            {/* Interactive Tabs Navigation */}
            <div className="war-room-tabs-bar" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === "directives"}
                className={`war-room-tab ${activeTab === "directives" ? "active" : ""}`}
                onClick={() => setActiveTab("directives")}
              >
                <FileText size={14} /> ICS-201 Directives ({councilData.executive_directives.length})
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === "transcript"}
                className={`war-room-tab ${activeTab === "transcript" ? "active" : ""}`}
                onClick={() => setActiveTab("transcript")}
              >
                <Radio size={14} /> Live Debate Transcript ({councilData.debate_transcript.length})
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === "friction"}
                className={`war-room-tab ${activeTab === "friction" ? "active" : ""}`}
                onClick={() => setActiveTab("friction")}
              >
                <Scale size={14} /> Friction &amp; Trade-offs
              </button>
            </div>

            {/* Tab 1: Executive Directives */}
            {activeTab === "directives" && (
              <div className="tab-pane directives-pane">
                <div className="directives-intro">
                  <strong>Formulated Executive Incident Directives</strong>
                  <span>Synthesized consensus orders balancing public health survival and power grid stability.</span>
                </div>
                <div className="directives-list">
                  {councilData.executive_directives.map((dir) => (
                    <div key={dir.directive_id} className="directive-item">
                      <div className="directive-topline">
                        <span className="directive-code">{dir.directive_id}</span>
                        <h4>{dir.title}</h4>
                        <span className="directive-window"><Clock size={12} /> {dir.enforcement_window}</span>
                      </div>
                      <p className="directive-mandate">{dir.mandate}</p>
                      <div className="directive-meta-row">
                        <div className="affected-sectors">
                          <span>AFFECTED:</span>
                          {dir.affected_sectors.map((sec) => (
                            <span key={sec} className="sector-tag">{sec}</span>
                          ))}
                        </div>
                        <div className="mitigated-risk">
                          <ShieldCheck size={13} />
                          <span>{dir.mitigated_risk}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {councilData.accepted_tradeoffs.length > 0 && (
                  <div className="tradeoffs-box">
                    <span className="tradeoffs-title"><Scale size={14} /> RATIFIED CROSS-SECTOR COMPROMISES:</span>
                    <ul>
                      {councilData.accepted_tradeoffs.map((item, idx) => (
                        <li key={idx}><Check size={13} /> {item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Tab 2: Live Debate Transcript */}
            {activeTab === "transcript" && (
              <div className="tab-pane transcript-pane">
                <div className="transcript-flow">
                  {councilData.debate_transcript.map((turn) => {
                    const colorClass = getMemberColor(turn.speaker_id);
                    const isCommander = turn.speaker_id.includes("commander") || turn.stance === "EXECUTIVE_SYNTHESIS";
                    return (
                      <div key={turn.turn_number} className={`transcript-bubble bubble-${colorClass} ${isCommander ? "bubble-commander" : ""}`}>
                        <div className="bubble-header">
                          <div className="bubble-speaker-info">
                            <span className="turn-number">Turn {turn.turn_number}</span>
                            <strong>{turn.speaker_title}</strong>
                          </div>
                          <span className={`stance-badge stance-${turn.stance.toLowerCase()}`}>
                            {turn.stance.replace(/_/g, " ")}
                          </span>
                        </div>
                        <p className="bubble-argument">{turn.argument}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Tab 3: Friction Matrix */}
            {activeTab === "friction" && (
              <div className="tab-pane friction-pane">
                <div className="friction-intro">
                  <strong>Inter-Agency Friction Matrix</strong>
                  <span>Direct points of friction identified during multi-agent cross-examination.</span>
                </div>
                <div className="friction-grid">
                  {Object.entries(councilData.friction_matrix).map(([pair, conflicts]) => (
                    <div key={pair} className="friction-card">
                      <div className="friction-card-title">
                        <AlertTriangle size={15} />
                        <h4>{pair.replace(/_/g, " ")}</h4>
                      </div>
                      <ul>
                        {conflicts.map((conflict, i) => (
                          <li key={i}>{conflict}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Modal Footer */}
            <footer className="war-room-modal-footer">
              <div className="audit-seal">
                <ShieldCheck size={14} />
                <span>CRYPTOGRAPHIC AUDIT SEAL: <code>{councilData.verification_hash.slice(0, 16)}…</code></span>
              </div>
              <div className="audit-latency">
                <span>Latency: {councilData.latency_ms ?? 120} ms</span>
              </div>
            </footer>
          </div>
        )}
      </section>
    </div>
  );
}
