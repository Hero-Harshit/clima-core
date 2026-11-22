import type { StyleSpecification } from "maplibre-gl";

const createSatelliteMapStyle = (saturation: number, contrast: number): StyleSpecification => ({
  version: 8,
  sources: {
    satellite: {
      type: "raster",
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
      maxzoom: 19,
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#edf5e9" } },
    {
      id: "satellite-imagery",
      type: "raster",
      source: "satellite",
      paint: { "raster-opacity": 0.96, "raster-saturation": saturation, "raster-contrast": contrast },
    },
  ],
});

export const pageBackdropStyle = createSatelliteMapStyle(-0.4, 0.02);
export const detailedSatelliteStyle = createSatelliteMapStyle(0.08, 0.08);

export const detailedStreetStyle: StyleSpecification = {
  version: 8,
  sources: {
    street: {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://d.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: "Map data (c) OpenStreetMap contributors, (c) CARTO",
      maxzoom: 19,
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#f8fafc" } },
    {
      id: "street-imagery",
      type: "raster",
      source: "street",
      paint: { "raster-opacity": 1.0 },
    },
  ],
};
