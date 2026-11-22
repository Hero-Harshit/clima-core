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
    assert.ok(cropData.provider.includes("gemini") || cropData.provider.includes("deterministic_engine"));
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
    assert.ok(examData.provider.includes("gemini") || examData.provider.includes("deterministic_engine"));
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
});

test("Utility Tools Modals: Component templates match design language and contains zero em dashes", () => {
  const cropModalPath = path.resolve(__dirname, "../src/CropHarvestModal.tsx");
  const examModalPath = path.resolve(__dirname, "../src/MonsoonExamModal.tsx");

  const cropContent = fs.readFileSync(cropModalPath, "utf-8");
  const examContent = fs.readFileSync(examModalPath, "utf-8");

  // Botanical design elements
  assert.ok(cropContent.includes("utility-modal-scrim"), "CropHarvestModal must use utility-modal-scrim");
  assert.ok(cropContent.includes("utility-modal-window"), "CropHarvestModal must use utility-modal-window");
  assert.ok(cropContent.includes("utility-status-grid"), "CropHarvestModal must render utility-status-grid");
  assert.ok(cropContent.includes("utility-ai-box"), "CropHarvestModal must render utility-ai-box");

  assert.ok(examContent.includes("utility-modal-scrim"), "MonsoonExamModal must use utility-modal-scrim");
  assert.ok(examContent.includes("utility-modal-window"), "MonsoonExamModal must use utility-modal-window");
  assert.ok(examContent.includes("utility-status-grid"), "MonsoonExamModal must render utility-status-grid");
  assert.ok(examContent.includes("utility-ai-box"), "MonsoonExamModal must render utility-ai-box");

  // Strict zero em dash check
  const emDashRegex = /[\u2014]/;
  assert.equal(emDashRegex.test(cropContent), false, "CropHarvestModal.tsx must contain zero em dashes");
  assert.equal(emDashRegex.test(examContent), false, "MonsoonExamModal.tsx must contain zero em dashes");
});
