import { useLayoutEffect, useRef, useState } from "react";
import Map from "react-map-gl/maplibre";
import type { PaddingOptions } from "maplibre-gl";
import { pageBackdropStyle } from "./satelliteMapStyles";

type MapBackdropProps = {
  latitude: number;
  longitude: number;
};

function MapBackdrop({ latitude, longitude }: MapBackdropProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [padding, setPadding] = useState<PaddingOptions>();

  useLayoutEffect(() => {
    const appShell = containerRef.current?.parentElement;
    if (!appShell) return;

    const updatePadding = () => {
      const bottom = Math.max(0, appShell.scrollHeight - window.innerHeight);
      setPadding((current) => current?.bottom === bottom ? current : { top: 0, right: 0, bottom, left: 0 });
    };

    updatePadding();
    const observer = new ResizeObserver(updatePadding);
    observer.observe(appShell);
    window.addEventListener("resize", updatePadding);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updatePadding);
    };
  }, []);

  return (
    <div className="map-page-backdrop" ref={containerRef}>
      <Map
        initialViewState={{ latitude, longitude, zoom: 10, padding }}
        latitude={latitude}
        longitude={longitude}
        zoom={10}
        padding={padding}
        mapStyle={pageBackdropStyle}
        interactive={false}
        attributionControl={false}
        pixelRatio={Math.min(window.devicePixelRatio || 1, 1.5)}
        maxCanvasSize={[4096, 4096]}
      />
    </div>
  );
}

export default MapBackdrop;
