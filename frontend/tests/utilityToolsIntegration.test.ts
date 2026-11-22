import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test("Utility Tools Integration: Live API endpoints return valid schema and metrics", async () => {
  const backendBase = "http://127.0.0.1:8000";

  // Test Crop Spoilage endpoint
  try {
    const cropRes = await fetch(`${backendBase}/api/tools/crop-spoilage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: "Delhi",
        crop_type: "paddy",
        harvest_stage: "mature_standing",
        storage_type: "traditional_tarpaulin",
      }),
    });

    assert.equal(cropRes.status, 200, "Crop Spoilage endpoint must respond with 200 OK");
    const cropData = await cropRes.json();
    assert.equal(cropData.crop_type, "paddy");
    assert.equal(cropData.harvest_stage, "mature_standing");
    assert.ok(typeof cropData.spoilage_risk_score === "number");
    assert.ok(typeof cropData.estimated_field_moisture_pct === "number");
    assert.ok(Array.isArray(cropData.drying_recommendations) && cropData.drying_recommendations.length > 0);
    assert.ok(Array.isArray(cropData.stages_breakdown) && cropData.stages_breakdown.length === 4);
    assert.ok(cropData.provider.includes("ClimaCore") || cropData.provider.includes("gemini") || cropData.provider.includes("deterministic_engine"));
  } catch (err) {
    assert.fail(`Crop Spoilage API request failed: ${err}`);
  }

  // Test Monsoon Exam Commute endpoint
  try {
    const examRes = await fetch(`${backendBase}/api/tools/monsoon-exam`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: "Delhi",
        exam_shift: "morning",
        transit_mode: "public_bus_metro",
        buffer_level: "moderate",
      }),
    });

    assert.equal(examRes.status, 200, "Monsoon Exam Commute endpoint must respond with 200 OK");
    const examData = await examRes.json();
    assert.equal(examData.exam_shift, "morning");
    assert.equal(examData.transit_mode, "public_bus_metro");
    assert.ok(typeof examData.commute_risk_score === "number");
    assert.ok(typeof examData.recommended_departure_advance_mins === "number");
    assert.ok(Array.isArray(examData.inundation_zones) && examData.inundation_zones.length === 3);
    assert.ok(Array.isArray(examData.safe_arrival_checklist) && examData.safe_arrival_checklist.length > 0);
    assert.ok(examData.provider.includes("ClimaCore") || examData.provider.includes("gemini") || examData.provider.includes("deterministic_engine"));
  } catch (err) {
    assert.fail(`Monsoon Exam API request failed: ${err}`);
  }

  // Test Factory Tin Roof endpoint
  try {
    const roofRes = await fetch(`${backendBase}/api/tools/factory-tin-roof`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: "Pune",
        roof_type: "uninsulated_corrugated_tin",
        shopfloor_area_sqm: 2500,
        hvac_capacity_tr: 150,
      }),
    });
    assert.equal(roofRes.status, 200, "Factory Tin Roof endpoint must respond with 200 OK");
    const roofData = await roofRes.json();
    assert.ok(typeof roofData.estimated_tin_surface_temp_c === "number");
    assert.ok(typeof roofData.shopfloor_wbgt_c === "number");
    assert.ok(typeof roofData.worker_heat_strain_risk === "string");
  } catch (err) {
    assert.fail(`Factory Tin Roof API request failed: ${err}`);
  }

  // Test Runway Crosswind endpoint
  try {
    const rwyRes = await fetch(`${backendBase}/api/tools/runway-crosswind`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        airport_code_or_city: "BOM",
        runway_heading_deg: 270,
        aircraft_category: "code_c_narrowbody_a320_b737",
      }),
    });
    assert.equal(rwyRes.status, 200, "Runway Crosswind endpoint must respond with 200 OK");
    const rwyData = await rwyRes.json();
    assert.ok(typeof rwyData.crosswind_component_knots === "number");
    assert.ok(typeof rwyData.crosswind_safety_status === "string");
  } catch (err) {
    assert.fail(`Runway Crosswind API request failed: ${err}`);
  }

  // Test Water Turbidity endpoint
  try {
    const waterRes = await fetch(`${backendBase}/api/tools/water-turbidity`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: "Delhi",
        plant_capacity_mld: 100,
        raw_water_source: "surface_river_run_of_the_river",
      }),
    });
    assert.equal(waterRes.status, 200, "Water Turbidity endpoint must respond with 200 OK");
    const waterData = await waterRes.json();
    assert.ok(typeof waterData.predicted_raw_turbidity_ntu === "number");
    assert.ok(typeof waterData.coagulant_dose_ppm === "number");
  } catch (err) {
    assert.fail(`Water Turbidity API request failed: ${err}`);
  }

  // Test Parametric Trigger endpoint
  try {
    const triggerRes = await fetch(`${backendBase}/api/tools/parametric-trigger`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: "Ahmedabad",
        peril_type: "excess_rainfall_flood",
        coverage_limit_inr_crores: 25.0,
      }),
    });
    assert.equal(triggerRes.status, 200, "Parametric Trigger endpoint must respond with 200 OK");
    const triggerData = await triggerRes.json();
    assert.ok(typeof triggerData.strike_breach_probability_pct === "number");
    assert.ok(typeof triggerData.trigger_status === "string");
  } catch (err) {
    assert.fail(`Parametric Trigger API request failed: ${err}`);
  }

  // Test Cold Storage Defense endpoint
  try {
    const coldRes = await fetch(`${backendBase}/api/tools/cold-storage-defense`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: "Kolkata",
        commodity_type: "vaccines_pharmaceuticals_2_8c",
        facility_capacity_metric_tons: 500,
      }),
    });
    assert.equal(coldRes.status, 200, "Cold Storage Defense endpoint must respond with 200 OK");
    const coldData = await coldRes.json();
    assert.ok(typeof coldData.passive_thermal_flywheel_hours === "number");
    assert.ok(typeof coldData.blackout_defense_readiness_level === "string");
  } catch (err) {
    assert.fail(`Cold Storage Defense API request failed: ${err}`);
  }

  // Test Mandi Commodities endpoint
  try {
    const commoditiesRes = await fetch(`${backendBase}/api/tools/mandi-commodities?state=Maharashtra`);
    assert.equal(commoditiesRes.status, 200, "Mandi Commodities endpoint must respond with 200 OK");
    const commoditiesData = await commoditiesRes.json();
    assert.equal(commoditiesData.state, "Maharashtra");
    assert.ok(Array.isArray(commoditiesData.commodities));
    assert.ok(commoditiesData.commodities.length > 0, "Must return available commodities for Maharashtra");
    assert.ok(commoditiesData.provider.includes("ClimaCore"));
  } catch (err) {
    assert.fail(`Mandi Commodities API request failed: ${err}`);
  }

  // Test Mandi Prices endpoint
  try {
    const mandiPricesRes = await fetch(`${backendBase}/api/tools/mandi-prices`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        commodity: "Tomato",
        state: "Maharashtra",
        district: "Pune",
      }),
    });
    assert.equal(mandiPricesRes.status, 200, "Mandi Prices endpoint must respond with 200 OK");
    const mandiPricesData = await mandiPricesRes.json();
    assert.equal(mandiPricesData.commodity, "Tomato");
    assert.equal(mandiPricesData.state, "Maharashtra");
    assert.ok(typeof mandiPricesData.avg_modal_price_kg === "number");
    assert.ok(typeof mandiPricesData.avg_modal_price_quintal === "number");
    assert.ok(typeof mandiPricesData.price_spread_kg === "number");
    assert.ok(Array.isArray(mandiPricesData.records));
    assert.ok(mandiPricesData.records.length > 0, "Must return at least one mandi record");
    // Verify quintal vs kg math
    const firstRecord = mandiPricesData.records[0];
    assert.equal(
      Math.round(firstRecord.modal_price_kg * 100),
      firstRecord.modal_price_quintal,
      "Price in Rs/kg must equal price in Rs/quintal divided by 100"
    );
    assert.ok(mandiPricesData.ai_market_advisory.length > 0, "Must include ClimaCore market advisory");
  } catch (err) {
    assert.fail(`Mandi Prices API request failed: ${err}`);
  }
});

test("Utility Tools Modals: Component templates match design language and contains zero em dashes", () => {
  const cropModalPath = path.resolve(__dirname, "../src/CropHarvestModal.tsx");
  const examModalPath = path.resolve(__dirname, "../src/MonsoonExamModal.tsx");
  const mandiModalPath = path.resolve(__dirname, "../src/MandiPricesModal.tsx");

  const cropContent = fs.readFileSync(cropModalPath, "utf-8");
  const examContent = fs.readFileSync(examModalPath, "utf-8");
  const mandiContent = fs.readFileSync(mandiModalPath, "utf-8");

  // Botanical design elements
  assert.ok(cropContent.includes("utility-modal-scrim"), "CropHarvestModal must use utility-modal-scrim");
  assert.ok(cropContent.includes("utility-modal-window"), "CropHarvestModal must use utility-modal-window");
  assert.ok(cropContent.includes("utility-status-grid"), "CropHarvestModal must render utility-status-grid");
  assert.ok(cropContent.includes("utility-ai-box"), "CropHarvestModal must render utility-ai-box");

  assert.ok(examContent.includes("utility-modal-scrim"), "MonsoonExamModal must use utility-modal-scrim");
  assert.ok(examContent.includes("utility-modal-window"), "MonsoonExamModal must use utility-modal-window");
  assert.ok(examContent.includes("utility-status-grid"), "MonsoonExamModal must render utility-status-grid");
  assert.ok(examContent.includes("utility-ai-box"), "MonsoonExamModal must render utility-ai-box");

  assert.ok(mandiContent.includes("utility-modal-scrim"), "MandiPricesModal must use utility-modal-scrim");
  assert.ok(mandiContent.includes("utility-modal-window"), "MandiPricesModal must use utility-modal-window");
  assert.ok(mandiContent.includes("utility-status-grid"), "MandiPricesModal must render utility-status-grid");
  assert.ok(mandiContent.includes("utility-ai-box"), "MandiPricesModal must render utility-ai-box");
  assert.ok(mandiContent.includes("mandi-table"), "MandiPricesModal must render mandi-table");

  // Strict zero em dash check
  const emDashRegex = /[\u2014]/;
  assert.equal(emDashRegex.test(cropContent), false, "CropHarvestModal.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(examContent), false, "MonsoonExamModal.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(mandiContent), false, "MandiPricesModal.tsx must contain zero em dashes");
});

test("Assistant Command Tools UI: Navbar button, dropdown, and command modes are properly mounted", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const stylesPath = path.resolve(__dirname, "../src/styles.css");

  const appContent = fs.readFileSync(appPath, "utf-8");
  const stylesContent = fs.readFileSync(stylesPath, "utf-8");

  // UI trigger button and dropdown structure
  assert.ok(appContent.includes("assistant-cmd-trigger-btn"), "App.tsx must include assistant-cmd-trigger-btn");
  assert.ok(appContent.includes("assistant-cmd-dropdown"), "App.tsx must include assistant-cmd-dropdown");
  assert.ok(appContent.includes("ASSISTANT_COMMAND_MODES"), "App.tsx must define ASSISTANT_COMMAND_MODES");
  assert.ok(appContent.includes('prefix: "/plan "'), "App.tsx must configure /plan command mode");
  assert.ok(appContent.includes('prefix: "/whatif "'), "App.tsx must configure /whatif command mode");
  assert.ok(appContent.includes('prefix: "/compare "'), "App.tsx must configure /compare command mode");

  // CSS constraints
  assert.ok(stylesContent.includes(".assistant-cmd-trigger-btn"), "styles.css must style .assistant-cmd-trigger-btn");
  assert.ok(stylesContent.includes(".assistant-cmd-dropdown"), "styles.css must style .assistant-cmd-dropdown");
  assert.ok(stylesContent.includes("border: 1px solid #166534 !important;"), "Dropdown must have uniform thin dark green border");

  // Strict zero em dash check
  const emDashRegex = /[\u2014]/;
  assert.equal(emDashRegex.test(appContent), false, "App.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(stylesContent), false, "styles.css must contain zero em dashes");
});

test("Assistant Command Tools Backend Execution: /plan, /whatif, and /compare generate standard structured intelligence", async () => {
  const backendBase = "http://127.0.0.1:8000";

  // Test /plan command mode
  try {
    const planRes = await fetch(`${backendBase}/api/assistant/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "/plan logistics fleet heatwave",
        city: "Delhi",
        persona: "transport",
        history: [],
      }),
    });
    assert.equal(planRes.status, 200, "Assistant /plan must respond with 200 OK");
    const planData = await planRes.json();
    assert.ok(planData.response.includes("Phase 1: Immediate Action (T-0 to T+2h)"), "/plan must contain Phase 1");
    assert.ok(planData.response.includes("Phase 2: Operational Defense (T+2 to T+6h)"), "/plan must contain Phase 2");
    assert.ok(planData.response.includes("Phase 3: Recovery & Verification (T+6h+)"), "/plan must contain Phase 3");
    assert.ok(planData.response.includes("- [ ]"), "/plan must contain actionable checkmarks");
    assert.equal(/[\u2014]/.test(planData.response), false, "/plan response must contain zero em dashes");
  } catch (err) {
    assert.fail(`Assistant /plan request failed: ${err}`);
  }

  // Test /whatif command mode
  try {
    const whatifRes = await fetch(`${backendBase}/api/assistant/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "/whatif +100mm cloudburst",
        city: "Mumbai",
        persona: "general",
        history: [],
      }),
    });
    assert.equal(whatifRes.status, 200, "Assistant /whatif must respond with 200 OK");
    const whatifData = await whatifRes.json();
    assert.ok(whatifData.response.includes("Drainage Capacity Breach Time"), "/whatif must include Drainage Capacity Breach Time");
    assert.ok(whatifData.response.includes("Underpass Runoff Depth"), "/whatif must include Underpass Runoff Depth");
    assert.ok(whatifData.response.includes("Fleet Delay Advance"), "/whatif must include Fleet Delay Advance");
    assert.ok(whatifData.response.includes("Mandi Arrival Impact"), "/whatif must include Mandi Arrival Impact");
    assert.equal(/[\u2014]/.test(whatifData.response), false, "/whatif response must contain zero em dashes");
  } catch (err) {
    assert.fail(`Assistant /whatif request failed: ${err}`);
  }

  // Test /compare command mode
  try {
    const compareRes = await fetch(`${backendBase}/api/assistant/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "/compare Mumbai Pune Tomato",
        city: "Mumbai",
        persona: "general",
        history: [],
      }),
    });
    assert.equal(compareRes.status, 200, "Assistant /compare must respond with 200 OK");
    const compareData = await compareRes.json();
    assert.ok(compareData.response.includes("Compound Risk Score"), "/compare must include Compound Risk Score");
    assert.ok(compareData.response.includes("Disruption Level"), "/compare must include Disruption Level");
    assert.ok(compareData.response.includes("Key Bottleneck"), "/compare must include Key Bottleneck");
    assert.ok(compareData.response.includes("Mandi Price"), "/compare must include Mandi Price");
    assert.equal(/[\u2014]/.test(compareData.response), false, "/compare response must contain zero em dashes");
  } catch (err) {
    assert.fail(`Assistant /compare request failed: ${err}`);
  }
});

test("Track Criteria Features: 1-Click Harvest & Sell Verdict, Yield Risk Badge, and Emergency Relief Convoy", async () => {
  const fs = await import("fs");
  const path = await import("path");

  const appPath = path.resolve(process.cwd(), "src/App.tsx");
  const mandiModalPath = path.resolve(process.cwd(), "src/MandiPricesModal.tsx");
  const corridorModalPath = path.resolve(process.cwd(), "src/CorridorRadarModal.tsx");
  const stylesPath = path.resolve(process.cwd(), "src/styles.css");

  const appContent = fs.readFileSync(appPath, "utf-8");
  const mandiContent = fs.readFileSync(mandiModalPath, "utf-8");
  const corridorContent = fs.readFileSync(corridorModalPath, "utf-8");
  const stylesContent = fs.readFileSync(stylesPath, "utf-8");

  // Feature 1: 1-Click Harvest & Sell Verdict
  assert.ok(mandiContent.includes("mandi-verdict-card"), "MandiPricesModal must render .mandi-verdict-card");
  assert.ok(mandiContent.includes("1-CLICK HARVEST & SELL VERDICT"), "MandiPricesModal must feature 1-Click Harvest & Sell Verdict title");
  assert.ok(appContent.includes("farmer-verdict-banner"), "App.tsx farmer briefing must include .farmer-verdict-banner");
  assert.ok(appContent.includes("Compare APMC Mandi Rates & Lock Sell Window") || appContent.includes("View Live Mandi Rates") || appContent.includes("setIsMandiPricesOpen(true)"), "Farmer briefing must include CTA button to open Mandi tool");

  // Feature 2: Projected Yield Risk Badge & Metric
  assert.ok(appContent.includes("yield-risk-strip"), "App.tsx agricultural intel must include .yield-risk-strip");
  assert.ok(appContent.includes("yield-risk-badge"), "App.tsx agricultural intel must include .yield-risk-badge");
  assert.ok(appContent.includes("Projected Yield Risk:"), "App.tsx must display Projected Yield Risk metric");

  // Feature 3: Emergency Relief & Medical Supply Convoy Mode
  assert.ok(corridorContent.includes("emergency_relief"), "CorridorRadarModal must include emergency_relief vehicle option");
  assert.ok(corridorContent.includes("Coordinates relief supplies & safe deliveries during floods"), "CorridorRadarModal must describe flood relief convoy capability");
  assert.ok(corridorContent.includes("convoy-priority-banner"), "CorridorRadarModal must render .convoy-priority-banner");

  // CSS assertions
  assert.ok(stylesContent.includes(".mandi-verdict-card"), "styles.css must style .mandi-verdict-card");
  assert.ok(stylesContent.includes(".farmer-verdict-banner"), "styles.css must style .farmer-verdict-banner");
  assert.ok(stylesContent.includes(".yield-risk-strip"), "styles.css must style .yield-risk-strip");
  assert.ok(stylesContent.includes(".yield-risk-badge"), "styles.css must style .yield-risk-badge");
  assert.ok(stylesContent.includes(".convoy-priority-banner"), "styles.css must style .convoy-priority-banner");

  // Strict zero em dash constraint
  const emDashRegex = /[\u2014]/;
  assert.equal(emDashRegex.test(appContent), false, "App.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(mandiContent), false, "MandiPricesModal.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(corridorContent), false, "CorridorRadarModal.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(stylesContent), false, "styles.css must contain zero em dashes");
});

test("Delivery Corridor Radar Backend: Emergency Relief & Medical Supply Convoy routing", async () => {
  const backendBase = "http://127.0.0.1:8000";
  try {
    const res = await fetch(`${backendBase}/api/tools/corridor-radar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        origin_city: "Mumbai",
        destination_city: "Pune",
        vehicle_type: "emergency_relief",
        cargo_sensitivity: "ambient_dry",
      }),
    });
    assert.equal(res.status, 200, "Corridor Radar emergency relief must respond with 200 OK");
    const data = await res.json();
    assert.equal(data.vehicle_type, "emergency_relief");
    assert.ok(/emergency/i.test(data.ai_insights) || /relief/i.test(data.ai_insights) || /priority/i.test(data.ai_insights), "Insights must address emergency relief convoy");
    assert.ok(Array.isArray(data.segments) && data.segments.length > 0, "Corridor Radar must return segments array");
  } catch (err) {
    assert.fail(`Corridor Radar emergency relief request failed: ${err}`);
  }
});

test("Google Calendar Smart Planner (Stage 1.5): Navbar button, dropdown, and 1-click intent URLs", async () => {
  const fs = await import("fs");
  const path = await import("path");

  const appPath = path.resolve(process.cwd(), "src/App.tsx");
  const stylesPath = path.resolve(process.cwd(), "src/styles.css");

  const appContent = fs.readFileSync(appPath, "utf-8");
  const stylesContent = fs.readFileSync(stylesPath, "utf-8");

  // UI presence in App.tsx
  assert.ok(appContent.includes("assistant-calendar-trigger-btn"), "App.tsx must include .assistant-calendar-trigger-btn");
  assert.ok(appContent.includes("assistant-calendar-dropdown"), "App.tsx must include .assistant-calendar-dropdown");
  assert.ok(appContent.includes("CLIMACORE SCHEDULE PLANNER"), "App.tsx must feature CLIMACORE SCHEDULE PLANNER header");
  assert.ok(appContent.includes("Schedule Safe Transit Window"), "App.tsx must offer Schedule Safe Transit Window action");
  assert.ok(appContent.includes("Add Active Severe Weather Advisory"), "App.tsx must offer Add Active Severe Weather Advisory action");
  assert.ok(appContent.includes("Open Google Calendar Workspace"), "App.tsx must offer Open Google Calendar Workspace action");
  assert.ok(appContent.includes("https://calendar.google.com/calendar/render"), "App.tsx must construct Google Calendar intent URLs");

  // CSS constraints
  assert.ok(stylesContent.includes(".assistant-calendar-trigger-btn"), "styles.css must style .assistant-calendar-trigger-btn");
  assert.ok(stylesContent.includes(".assistant-calendar-dropdown"), "styles.css must style .assistant-calendar-dropdown");
  assert.ok(stylesContent.includes(".assistant-calendar-city-badge"), "styles.css must style .assistant-calendar-city-badge");

  // Strict zero em dash constraint
  const emDashRegex = /[\u2014]/;
  assert.equal(emDashRegex.test(appContent), false, "App.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(stylesContent), false, "styles.css must contain zero em dashes");
});
