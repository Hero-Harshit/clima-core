import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";

export type UtilitySelectOption = {
  id: string;
  label: string;
  desc?: string;
};

export type UtilityCustomSelectProps = {
  id?: string;
  value: string;
  options: UtilitySelectOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
  ariaLabel?: string;
  placeholder?: string;
};

export default function UtilityCustomSelect({
  id: propId,
  value,
  options,
  onChange,
  disabled = false,
  ariaLabel,
  placeholder = "Select an option",
}: UtilityCustomSelectProps) {
  const generatedId = useId();
  const selectId = propId || generatedId;
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);

  const selectedIndex = Math.max(0, options.findIndex((opt) => opt.id === value));
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const selectedOption = options.find((opt) => opt.id === value);

  useEffect(() => {
    setActiveIndex(selectedIndex);
  }, [selectedIndex]);

  useEffect(() => {
    if (!isOpen) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    menuRef.current
      ?.querySelector<HTMLElement>(`[data-option-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, isOpen]);

  const selectOption = (opt: UtilitySelectOption) => {
    onChange(opt.id);
    setIsOpen(false);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) return;

    if (event.key === "Escape" && isOpen) {
      event.preventDefault();
      setIsOpen(false);
    } else if (event.key === "Tab") {
      setIsOpen(false);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!isOpen) {
        setActiveIndex(selectedIndex);
        setIsOpen(true);
      } else if (options.length) {
        const offset = event.key === "ArrowDown" ? 1 : -1;
        setActiveIndex((idx) => (idx + offset + options.length) % options.length);
      }
    } else if (event.key === "Home" || event.key === "End") {
      if (!isOpen || !options.length) return;
      event.preventDefault();
      setActiveIndex(event.key === "Home" ? 0 : options.length - 1);
    } else if ((event.key === "Enter" || event.key === " ") && isOpen) {
      event.preventDefault();
      const opt = options[activeIndex];
      if (opt) selectOption(opt);
    }
  };

  return (
    <div className={`utility-custom-select ${isOpen ? "open" : ""}`} ref={rootRef}>
      <button
        type="button"
        id={selectId}
        className="utility-custom-select-trigger"
        role="combobox"
        aria-label={ariaLabel || "Select dropdown"}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={`utility-select-list-${selectId}`}
        aria-activedescendant={isOpen ? `utility-option-${selectId}-${activeIndex}` : undefined}
        disabled={disabled}
        onClick={() => {
          if (disabled) return;
          setActiveIndex(selectedIndex);
          setIsOpen((open) => !open);
        }}
        onKeyDown={handleKeyDown}
      >
        <span className={`utility-custom-select-value ${!selectedOption ? "is-placeholder" : ""}`}>
          {selectedOption?.label ?? placeholder}
        </span>
        <ChevronDown size={15} className="utility-select-chevron" aria-hidden="true" />
      </button>

      {isOpen && (
        <div
          className="utility-custom-select-menu"
          id={`utility-select-list-${selectId}`}
          role="listbox"
          ref={menuRef}
        >
          {options.map((opt, index) => {
            const isSelected = opt.id === value;
            const isActive = activeIndex === index;
            return (
              <div
                key={opt.id}
                id={`utility-option-${selectId}-${index}`}
                className={`utility-custom-select-option ${isSelected ? "selected" : ""} ${isActive ? "active" : ""}`}
                data-option-index={index}
                role="option"
                aria-selected={isSelected}
                onMouseDown={(e) => e.preventDefault()}
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => selectOption(opt)}
              >
                <div className="utility-option-text-wrap">
                  <div className="utility-option-label">{opt.label}</div>
                  {opt.desc && <div className="utility-option-desc">{opt.desc}</div>}
                </div>
                {isSelected && <Check size={14} className="utility-option-check" aria-hidden="true" />}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
