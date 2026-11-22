import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { en } from "../src/i18n/locales/en.ts";
import { hi } from "../src/i18n/locales/hi.ts";
import { bn } from "../src/i18n/locales/bn.ts";
import { ta } from "../src/i18n/locales/ta.ts";
import { te } from "../src/i18n/locales/te.ts";
import { mr } from "../src/i18n/locales/mr.ts";
import { gu } from "../src/i18n/locales/gu.ts";
import { getTranslations, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE } from "../src/i18n/index.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function getDeepKeys(obj: Record<string, unknown>, prefix = ""): string[] {
  let keys: string[] = [];
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      keys = keys.concat(getDeepKeys(value as Record<string, unknown>, fullKey));
    } else {
      keys.push(fullKey);
    }
  }
  return keys.sort();
}

test("i18n: Supported languages specification", () => {
  const codes = SUPPORTED_LANGUAGES.map((l) => l.code);
  assert.deepEqual(codes, ["en", "hi", "bn", "ta", "te", "mr", "gu"]);
  assert.equal(DEFAULT_LANGUAGE, "en");
});

test("i18n: Fallback to English for unknown language code", () => {
  const fallback = getTranslations("fr");
  assert.equal(fallback.topbar.brandTitle, en.topbar.brandTitle);
  assert.equal(fallback.hero.title, en.hero.title);
});

test("i18n: Strict key parity between English and all Indic locales", () => {
  const enKeys = getDeepKeys(en as unknown as Record<string, unknown>);
  const indicLocales = [
    { code: "hi", dict: hi },
    { code: "bn", dict: bn },
    { code: "ta", dict: ta },
    { code: "te", dict: te },
    { code: "mr", dict: mr },
    { code: "gu", dict: gu },
  ];

  for (const { code, dict } of indicLocales) {
    const localeKeys = getDeepKeys(dict as unknown as Record<string, unknown>);
    assert.deepEqual(
      localeKeys,
      enKeys,
      `Locale '${code}' key set does not match canonical English key set`
    );

    // Ensure no values are empty strings or undefined
    for (const keyPath of enKeys) {
      const parts = keyPath.split(".");
      let val: unknown = dict;
      for (const part of parts) {
        val = (val as Record<string, unknown>)[part];
      }
      assert.notEqual(val, undefined, `Locale '${code}' key '${keyPath}' is undefined`);
      assert.notEqual(val, "", `Locale '${code}' key '${keyPath}' is empty`);
      if (Array.isArray(val)) {
        assert.ok(val.length > 0, `Locale '${code}' array '${keyPath}' is empty`);
        for (const item of val) {
          assert.equal(typeof item, "string", `Locale '${code}' array item in '${keyPath}' is not string`);
          assert.ok(item.length > 0, `Locale '${code}' array item in '${keyPath}' is empty`);
        }
      } else {
        assert.equal(typeof val, "string", `Locale '${code}' key '${keyPath}' is not a string`);
      }
    }
  }
});

test("i18n: Script adherence for all 6 Indic languages", () => {
  const scriptRegexes: Record<string, RegExp> = {
    hi: /[\u0900-\u097F]/, // Devanagari
    mr: /[\u0900-\u097F]/, // Devanagari
    bn: /[\u0980-\u09FF]/, // Bengali
    ta: /[\u0B80-\u0BFF]/, // Tamil
    te: /[\u0C00-\u0C7F]/, // Telugu
    gu: /[\u0A80-\u0AFF]/, // Gujarati
  };

  const indicLocales = [
    { code: "hi", dict: hi },
    { code: "bn", dict: bn },
    { code: "ta", dict: ta },
    { code: "te", dict: te },
    { code: "mr", dict: mr },
    { code: "gu", dict: gu },
  ];

  for (const { code, dict } of indicLocales) {
    const regex = scriptRegexes[code];
    assert.ok(regex, `Missing regex for ${code}`);

    // Verify key UI elements contain native script characters
    assert.match(dict.topbar.brandSubtitle, regex, `${code} topbar.brandSubtitle should have native script`);
    assert.match(dict.hero.title, regex, `${code} hero.title should have native script`);
    assert.match(dict.hero.viewIn, regex, `${code} hero.viewIn should have native script`);
    assert.match(dict.geospatial.kicker, regex, `${code} geospatial.kicker should have native script`);
    assert.match(dict.personaCard.cardLabel, regex, `${code} personaCard.cardLabel should have native script`);
    assert.match(dict.forecast.kicker, regex, `${code} forecast.kicker should have native script`);
    assert.match(dict.environmentalCards.cardFloodTitle, regex, `${code} environmentalCards.cardFloodTitle should have native script`);
    assert.match(dict.cascadingImpact.heading, regex, `${code} cascadingImpact.heading should have native script`);
    assert.match(dict.modelConfidence.conformalCoverage, regex, `${code} modelConfidence.conformalCoverage should have native script`);
    assert.match(dict.footer.linkPrivacy, regex, `${code} footer.linkPrivacy should have native script`);
  }
});

test("i18n: Policy and modal subpages retain English text", () => {
  const srcDir = path.join(__dirname, "..", "src");

  // Read policy modal components and documentation sources
  const filesToCheck = [
    "PrivacyPolicy.tsx",
    "AboutUs.tsx",
    "FAQ.tsx",
    "methodology.md",
    "data-sources.md",
  ];

  for (const filename of filesToCheck) {
    const filePath = path.join(srcDir, filename);
    if (!fs.existsSync(filePath)) continue;

    const content = fs.readFileSync(filePath, "utf-8");
    // Ensure English headings/sentences exist in policy files
    assert.ok(
      content.includes("Clima-Core") || content.includes("Clima Core") || content.includes("Policy") || content.includes("Data") || content.includes("Model"),
      `${filename} should contain English content`
    );

    // Verify that policy documents don't have non-Latin translations injected into their core text
    const devanagariCount = (content.match(/[\u0900-\u097F]/g) || []).length;
    const tamilCount = (content.match(/[\u0B80-\u0BFF]/g) || []).length;
    const teluguCount = (content.match(/[\u0C00-\u0C7F]/g) || []).length;
    const bengaliCount = (content.match(/[\u0980-\u09FF]/g) || []).length;
    const gujaratiCount = (content.match(/[\u0A80-\u0AFF]/g) || []).length;

    assert.equal(
      devanagariCount + tamilCount + teluguCount + bengaliCount + gujaratiCount,
      0,
      `${filename} must remain strictly in English without Indic translations`
    );
  }
});
