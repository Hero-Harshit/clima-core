import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test("Hero layout: App.tsx wraps threat banner and persona tools side by side", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const content = fs.readFileSync(appPath, "utf-8");

  assert.ok(
    content.includes("hero-cards-row is-split"),
    "App.tsx must contain hero-cards-row is-split wrapper for side by side layout"
  );
  assert.ok(
    content.includes("threat-banner"),
    "App.tsx must contain threat-banner inside hero-cards-row"
  );
  assert.ok(
    content.includes("persona-tools-card"),
    "App.tsx must contain persona-tools-card unconditionally inside hero-cards-row"
  );
  assert.ok(
    content.includes("t.personaTools.defaultTitle"),
    "App.tsx must show default title when no occupation is selected"
  );
  assert.ok(
    content.includes("t.personaTools.awaitingPrompt"),
    "App.tsx must show awaiting prompt instructing user to select an occupation"
  );
  assert.ok(
    !content.includes("t.threatBanner.pillAwaiting"),
    "App.tsx must not render awaiting selection pill in threat banner"
  );
  assert.ok(
    !content.includes("t.personaTools.badgeAwaiting"),
    "App.tsx must not render awaiting selection pill in utility tools card"
  );
  assert.ok(
    !content.includes("persona-tool-action-pill"),
    "App.tsx must remove action pill with arrow from tool cards"
  );
  assert.ok(
    !content.includes("persona-tools-count-badge"),
    "App.tsx must not render active tools count badge in utility card"
  );
  assert.ok(
    content.includes("is-single-tool") && content.includes("is-multi-tool"),
    "App.tsx must differentiate single vs multi tool grid layouts"
  );
});

test("Navbar & Footer: Explore pages moved to navbar 3-dot menu and removed from footer", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const content = fs.readFileSync(appPath, "utf-8");

  assert.ok(
    !content.includes('className="home-footer"'),
    "App.tsx must not contain home-footer on the homepage"
  );
  assert.ok(
    content.includes("tools-menu-page-icon"),
    "App.tsx must render page icons in the navbar 3-dot menu"
  );
  assert.ok(
    content.includes('openNavigationPage("provenance")') &&
    content.includes('openNavigationPage("privacy")') &&
    content.includes('openNavigationPage("about")') &&
    content.includes('openNavigationPage("methodology")') &&
    content.includes('openNavigationPage("data-sources")') &&
    content.includes('openNavigationPage("model-performance")') &&
    content.includes('openNavigationPage("faq")'),
    "App.tsx navbar 3-dot menu must link to all 7 explore pages"
  );
});

test("Hero layout: styles.css contains side-by-side grid rules for is-split", () => {
  const cssPath = path.resolve(__dirname, "../src/styles.css");
  const content = fs.readFileSync(cssPath, "utf-8");

  assert.ok(
    content.includes(".weather-hero-region .hero-cards-row.is-split"),
    "styles.css must style .weather-hero-region .hero-cards-row.is-split"
  );
  assert.ok(
    content.includes("grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)"),
    "styles.css must define 2-column equal split for hero cards"
  );
  assert.ok(
    content.includes(".weather-hero-region .hero-cards-row.is-split .persona-tools-grid"),
    "styles.css must style persona-tools-grid"
  );
  assert.ok(
    content.includes("grid-template-columns: repeat(2, minmax(0, 1fr))"),
    "styles.css must define 2-column layout for utility tools"
  );
  assert.ok(
    content.includes(".weather-hero-region .hero-cards-row.is-split .threat-banner"),
    "styles.css must style threat-banner inside is-split"
  );
  assert.ok(
    content.includes(".weather-hero-region .hero-cards-row.is-split .persona-tools-card"),
    "styles.css must style persona-tools-card inside is-split"
  );
  assert.ok(
    content.includes(".weather-hero-region .hero-cards-row.is-split .persona-tools-empty.is-awaiting"),
    "styles.css must style persona-tools-empty.is-awaiting"
  );
  assert.ok(
    content.includes(".persona-tools-card") && content.includes("border: 1px solid var(--dark-green-card-border) !important"),
    "styles.css must style persona-tools-card with dark green card border"
  );
});

test("Code integrity: No em dash characters in modified frontend files", () => {
  const filesToCheck = [
    "../src/App.tsx",
    "../src/styles.css",
    "../src/i18n/types.ts",
    "../src/CropHarvestModal.tsx",
    "../src/MonsoonExamModal.tsx",
    "../src/MetroVsRoadModal.tsx",
    "../src/HeatStrokeSurgeModal.tsx",
    "../src/HospitalGridFuelModal.tsx",
    "../src/CoastalSquallModal.tsx",
    "../src/IceToFishRatioModal.tsx",
    "../src/FactoryTinRoofModal.tsx",
    "../src/RunwayCrosswindModal.tsx",
    "../src/WaterTurbidityModal.tsx",
    "../src/ParametricTriggerModal.tsx",
    "../src/ColdStorageModal.tsx",
  ];

  const emDashRegex = /[\u2014]/;
  for (const relPath of filesToCheck) {
    const fullPath = path.resolve(__dirname, relPath);
    const content = fs.readFileSync(fullPath, "utf-8");
    assert.equal(
      emDashRegex.test(content),
      false,
      `${relPath} must not contain any em dash characters`
    );
  }
});

test("Persona Utility Tools: Farmer, Student, Commuter, Healthcare and Maritime tools mapped and calibrated", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const content = fs.readFileSync(appPath, "utf-8");

  assert.ok(
    content.includes('id: "crop_spoilage"') && content.includes("cropHarvestTitle"),
    "App.tsx must map crop_spoilage tool with cropHarvestTitle for farmer persona"
  );
  assert.ok(
    content.includes('id: "monsoon_exam"') && content.includes("monsoonExamTitle"),
    "App.tsx must map monsoon_exam tool with monsoonExamTitle for student persona"
  );
  assert.ok(
    content.includes('id: "metro_vs_road"') && content.includes("metroVsRoadTitle"),
    "App.tsx must map metro_vs_road tool with metroVsRoadTitle for commuter persona"
  );
  assert.ok(
    content.includes('id: "heat_stroke_surge"') && content.includes("heatStrokeSurgeTitle"),
    "App.tsx must map heat_stroke_surge tool with heatStrokeSurgeTitle for healthcare persona"
  );
  assert.ok(
    content.includes('id: "hospital_grid_fuel"') && content.includes("hospitalGridFuelTitle"),
    "App.tsx must map hospital_grid_fuel tool with hospitalGridFuelTitle for healthcare persona"
  );
  assert.ok(
    content.includes('id: "coastal_squall"') && content.includes("coastalSquallTitle"),
    "App.tsx must map coastal_squall tool with coastalSquallTitle for maritime persona"
  );
  assert.ok(
    content.includes('id: "ice_to_fish_ratio"') && content.includes("iceToFishRatioTitle"),
    "App.tsx must map ice_to_fish_ratio tool with iceToFishRatioTitle for maritime persona"
  );
  assert.ok(
    content.includes('id: "factory_tin_roof"') && content.includes("factoryTinRoofTitle"),
    "App.tsx must map factory_tin_roof tool with factoryTinRoofTitle for industry persona"
  );
  assert.ok(
    content.includes('id: "runway_crosswind"') && content.includes("runwayCrosswindTitle"),
    "App.tsx must map runway_crosswind tool with runwayCrosswindTitle for aviation persona"
  );
  assert.ok(
    content.includes('id: "water_turbidity"') && content.includes("waterTurbidityTitle"),
    "App.tsx must map water_turbidity tool with waterTurbidityTitle for water persona"
  );
  assert.ok(
    content.includes('id: "parametric_trigger"') && content.includes("parametricTriggerTitle"),
    "App.tsx must map parametric_trigger tool with parametricTriggerTitle for insurance persona"
  );
  assert.ok(
    content.includes('id: "cold_storage_defense"') && content.includes("coldStorageTitle"),
    "App.tsx must map cold_storage_defense tool with coldStorageTitle for cold_chain persona"
  );
  assert.ok(
    content.includes("<CropHarvestModal") &&
    content.includes("<MonsoonExamModal") &&
    content.includes("<MetroVsRoadModal") &&
    content.includes("<HeatStrokeSurgeModal") &&
    content.includes("<HospitalGridFuelModal") &&
    content.includes("<CoastalSquallModal") &&
    content.includes("<IceToFishRatioModal") &&
    content.includes("<FactoryTinRoofModal") &&
    content.includes("<RunwayCrosswindModal") &&
    content.includes("<WaterTurbidityModal") &&
    content.includes("<ParametricTriggerModal") &&
    content.includes("<ColdStorageModal"),
    "App.tsx must render all tool modal overlays"
  );
});
