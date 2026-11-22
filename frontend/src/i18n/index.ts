import type { TranslationDictionary } from "./types.ts";
import { en } from "./locales/en.ts";
import { hi } from "./locales/hi.ts";
import { bn } from "./locales/bn.ts";
import { ta } from "./locales/ta.ts";
import { te } from "./locales/te.ts";
import { mr } from "./locales/mr.ts";
import { gu } from "./locales/gu.ts";

export type * from "./types.ts";

export const DEFAULT_LANGUAGE = "en";

export const SUPPORTED_LANGUAGES = [
  { code: "en", label: "English", nativeName: "English" },
  { code: "hi", label: "Hindi", nativeName: "हिन्दी" },
  { code: "bn", label: "Bengali", nativeName: "বাংলা" },
  { code: "ta", label: "Tamil", nativeName: "தமிழ்" },
  { code: "te", label: "Telugu", nativeName: "తెలుగు" },
  { code: "mr", label: "Marathi", nativeName: "मराठी" },
  { code: "gu", label: "Gujarati", nativeName: "ગુજરાતી" },
] as const;

export const LOCALES: Record<string, TranslationDictionary> = {
  en,
  hi,
  bn,
  ta,
  te,
  mr,
  gu,
};

export function getTranslations(langCode: string): TranslationDictionary {
  const normalized = (langCode || "en").toLowerCase();
  return LOCALES[normalized] || LOCALES["en"];
}
