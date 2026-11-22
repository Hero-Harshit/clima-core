import { useState, useMemo } from "react";
import {
  Search,
  ChevronDown,
  ChevronUp,
  CircleHelp,
  Filter,
  CheckCircle2,
  X,
} from "lucide-react";
import { FAQ_DATA, FAQ_CATEGORIES, FAQItem } from "./faqData";

export default function FAQ() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [openIds, setOpenIds] = useState<Set<number>>(new Set([1, 2, 6, 11, 16, 43]));

  const filteredFaqs = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return FAQ_DATA.filter((item) => {
      const matchesCategory =
        selectedCategory === "All" || item.category === selectedCategory;
      if (!matchesCategory) return false;
      if (!query) return true;
      return (
        item.question.toLowerCase().includes(query) ||
        item.answer.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query) ||
        String(item.id).includes(query)
      );
    });
  }, [searchQuery, selectedCategory]);

  const toggleItem = (id: number) => {
    setOpenIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const expandAll = () => {
    setOpenIds(new Set(filteredFaqs.map((f) => f.id)));
  };

  const collapseAll = () => {
    setOpenIds(new Set());
  };

  return (
    <section className="faq-page-container" aria-label="Frequently Asked Questions">
      <div className="faq-hero">
        <div className="faq-hero-badge">
          <CircleHelp size={15} />
          <span>50 FREQUENTLY ASKED QUESTIONS</span>
        </div>
        <h2 className="faq-title">Everything you need to know about ClimaCore</h2>
        <p className="faq-subtitle">
          Comprehensive, scientifically grounded answers covering our machine learning engines, physical process
          hydrology, uncertainty calibration, data sources, and operational decision support.
        </p>
      </div>

      <div className="faq-controls">
        <div className="faq-search-wrapper">
          <Search size={18} className="faq-search-icon" />
          <input
            type="text"
            className="faq-search-input"
            placeholder="Search all 50 questions, methods, models, or data sources..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              className="faq-search-clear"
              onClick={() => setSearchQuery("")}
              aria-label="Clear search"
            >
              <X size={15} />
            </button>
          )}
        </div>

        <div className="faq-actions-row">
          <div className="faq-categories">
            {FAQ_CATEGORIES.map((cat) => {
              const count =
                cat === "All"
                  ? FAQ_DATA.length
                  : FAQ_DATA.filter((f) => f.category === cat).length;
              return (
                <button
                  key={cat}
                  className={`faq-cat-pill ${selectedCategory === cat ? "active" : ""}`}
                  onClick={() => setSelectedCategory(cat)}
                  type="button"
                >
                  <span>{cat}</span>
                  <span className="faq-cat-count">{count}</span>
                </button>
              );
            })}
          </div>

          <div className="faq-bulk-toggle">
            <button className="faq-toggle-btn" onClick={expandAll} type="button">
              Expand all
            </button>
            <span className="faq-toggle-divider">/</span>
            <button className="faq-toggle-btn" onClick={collapseAll} type="button">
              Collapse all
            </button>
          </div>
        </div>
      </div>

      <div className="faq-stats-bar">
        <span>Showing <strong>{filteredFaqs.length}</strong> of <strong>50</strong> FAQs</span>
        {selectedCategory !== "All" && (
          <span className="faq-active-filter">
            <Filter size={12} /> Category: {selectedCategory}
          </span>
        )}
      </div>

      <div className="faq-list">
        {filteredFaqs.length === 0 ? (
          <div className="faq-empty-state">
            <CircleHelp size={36} />
            <h4>No matching questions found</h4>
            <p>Try searching for different keywords such as conformal, flood, heat, air quality, or reset your filter.</p>
            <button
              className="faq-reset-btn"
              onClick={() => {
                setSearchQuery("");
                setSelectedCategory("All");
              }}
              type="button"
            >
              Reset filters
            </button>
          </div>
        ) : (
          filteredFaqs.map((item: FAQItem) => {
            const isOpen = openIds.has(item.id);
            return (
              <article
                key={item.id}
                className={`faq-card ${isOpen ? "open" : ""}`}
              >
                <button
                  className="faq-card-header"
                  onClick={() => toggleItem(item.id)}
                  aria-expanded={isOpen}
                  type="button"
                >
                  <div className="faq-card-meta">
                    <span className="faq-id-badge">#{item.id}</span>
                    <span className="faq-category-tag">{item.category}</span>
                  </div>
                  <h3 className="faq-question-text">{item.question}</h3>
                  <div className="faq-chevron-wrap">
                    {isOpen ? <ChevronUp size={17} /> : <ChevronDown size={17} />}
                  </div>
                </button>

                {isOpen && (
                  <div className="faq-card-body">
                    <div className="faq-answer-content">
                      {item.answer.split("\n\n").map((paragraph, pIdx) => (
                        <p key={pIdx}>{paragraph}</p>
                      ))}
                    </div>
                  </div>
                )}
              </article>
            );
          })
        )}
      </div>

      <div className="faq-footer-note">
        <CheckCircle2 size={16} />
        <div>
          <strong>Scientific Accuracy Guarantee:</strong> All 50 responses reflect the operational physics, empirical
          benchmarks, and codebase implementation of the ClimaCore Environmental Intelligence Platform as of November 2026.
        </div>
      </div>
    </section>
  );
}
