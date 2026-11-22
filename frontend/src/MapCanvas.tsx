import Map, { Layer, Marker, Source } from "react-map-gl/maplibre";
import { detailedSatelliteStyle } from "./satelliteMapStyles";

type MapCanvasProps = {
  latitude: number;
  longitude: number;
};

function circleRing(longitude: number, latitude: number, radiusKm: number): number[][] {
  const earthRadiusKm = 6371;
  const angularDistance = radiusKm / earthRadiusKm;
  const centerLatitude = latitude * Math.PI / 180;
  const centerLongitude = longitude * Math.PI / 180;
  const coordinates: number[][] = [];

  for (let bearing = 0; bearing <= 360; bearing += 6) {
    const direction = bearing * Math.PI / 180;
    const pointLatitude = Math.asin(
      Math.sin(centerLatitude) * Math.cos(angularDistance)
      + Math.cos(centerLatitude) * Math.sin(angularDistance) * Math.cos(direction),
    );
    const pointLongitude = centerLongitude + Math.atan2(
      Math.sin(direction) * Math.sin(angularDistance) * Math.cos(centerLatitude),
      Math.cos(angularDistance) - Math.sin(centerLatitude) * Math.sin(pointLatitude),
    );
    coordinates.push([pointLongitude * 180 / Math.PI, pointLatitude * 180 / Math.PI]);
  }
  return coordinates;
}

function MapCanvas({ latitude, longitude }: MapCanvasProps) {
  return (
    <Map
      initialViewState={{ latitude, longitude, zoom: 10 }}
      longitude={longitude}
      latitude={latitude}
      zoom={10}
      mapStyle={detailedSatelliteStyle}
      scrollZoom={false}
      attributionControl={false}
    >
      <Source id="risk-rings" type="geojson" data={{ type: "FeatureCollection", features: [30, 15, 5].map((km) => ({ type: "Feature" as const, properties: { km }, geometry: { type: "Polygon" as const, coordinates: [circleRing(longitude, latitude, km)] } })) }}>
        <Layer id="rings-fill" type="fill" paint={{ "fill-color": "#16a34a", "fill-opacity": ["match", ["get", "km"], 5, 0.1, 15, 0.045, 30, 0.018, 0.03] }} />
        <Layer id="rings-outline" type="line" paint={{ "line-color": "#34d399", "line-opacity": 0.6, "line-width": 1.5 }} />
      </Source>
      <Marker longitude={longitude} latitude={latitude} anchor="center"><div className="map-marker"><span /><i /></div></Marker>
    </Map>
  );
}

export default MapCanvas;
