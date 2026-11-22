import { Component, ReactNode, useLayoutEffect, useRef, useState } from "react";
import Map from "react-map-gl/maplibre";
import type { PaddingOptions } from "maplibre-gl";
import { pageBackdropStyle } from "./satelliteMapStyles";

type MapBackdropProps = {
  latitude: number;
  longitude: number;
};

class MapErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: unknown) {
    console.warn("MapBackdrop caught rendering error, preserving UI layout:", error);
  }

  render() {
    if (this.state.hasError) {
      return null;
    }
    return this.props.children;
  }
}

function MapBackdropInner({ latitude, longitude }: MapBackdropProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [padding, setPadding] = useState<PaddingOptions>();

  useLayoutEffect(() => {
    const appShell = containerRef.current?.parentElement;
    if (!appShell) return;

    const updatePadding = () => {
      // MapLibre requires containerHeight > padding.top + padding.bottom
      // Clamping to a safe max prevents camera projection failures on tall layouts
      const viewportH = window.innerHeight || 800;
      const rawBottom = Math.max(0, appShell.scrollHeight - viewportH);
      const safeMax = Math.max(0, Math.min(100, Math.floor((viewportH - 240) / 4)));
      const bottom = Math.min(rawBottom, safeMax);
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

function MapBackdrop(props: MapBackdropProps) {
  return (
    <MapErrorBoundary>
      <MapBackdropInner {...props} />
    </MapErrorBoundary>
  );
}

export default MapBackdrop;
