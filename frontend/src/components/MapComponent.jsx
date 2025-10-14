// src/MapComponent.jsx
import React, { useState, useEffect } from "react";
import { Map, AdvancedMarker, InfoWindow, useMap } from "@vis.gl/react-google-maps";

const MAP_ID = "YOUR_MAP_ID";

/**
 * 지도 이동을 담당하는 별도의 컨트롤러 컴포넌트
 * useMap 훅은 Map 컴포넌트의 자식 요소에서만 사용할 수 있기 때문에 분리합니다.
 */
const MapController = ({ center }) => {
  const map = useMap();

  useEffect(() => {
    // map 인스턴스가 있고, center 값이 유효할 때 panTo(부드럽게 이동)를 호출합니다.
    if (map && center) {
      map.panTo(center);
    }
  }, [map, center]); // map 인스턴스나 center 값이 변경될 때만 실행됩니다.

  return null; // 이 컴포넌트는 화면에 아무것도 그리지 않습니다.
};


const MapComponent = ({ places, center }) => {
  const [selectedPlace, setSelectedPlace] = useState(null);

  return (
    <div className="map-container card">
      <Map
        style={{ width: "100%", height: "100%" }}
        // defaultCenter는 맨 처음 렌더링 시에만 사용되어 사용자 조작을 방해하지 않습니다.
        defaultCenter={center}
        defaultZoom={15}
        gestureHandling={"greedy"}
        disableDefaultUI={true}
        mapId={MAP_ID}
        onClick={() => setSelectedPlace(null)}
      >
        {/* 지도 컨트롤러 컴포넌트를 자식으로 추가하여 지도를 제어합니다. */}
        <MapController center={center} />

        {places.map((place) => (
          <AdvancedMarker
            key={place.id}
            position={place.location}
            onClick={() => setSelectedPlace(place)}
          />
        ))}

        {selectedPlace && (
          <InfoWindow
            position={selectedPlace.location}
            onCloseClick={() => setSelectedPlace(null)}
          >
            <div>
              <strong>{selectedPlace.displayName}</strong>
              <p>{selectedPlace.formattedAddress}</p>
            </div>
          </InfoWindow>
        )}
      </Map>
    </div>
  );
};

export default MapComponent;