<template>
  <!-- Map tooltip positioned using CSS and projected coordinates -->
  <div v-if="tooltip.object" :style="tooltipPositionStyle" class="map-tooltip">
    {{ tooltip.object.name }}
  </div>
</template>

<script setup lang="ts">
import { computed, inject, watch, onUnmounted, reactive, ref } from "vue";
import type { Ref, CSSProperties } from "vue";
import { PathLayer } from "@deck.gl/layers";
import { TangramApi, Entity, Disposable } from "@open-aviation/tangram-core/api";
import { AircraftState } from "../../../tangram_jet1090/src/tangram_jet1090/AircraftLayer.vue";
import { PathStyleExtension } from "@deck.gl/extensions";

const tangramApi = inject<TangramApi>("tangramApi");
if (!tangramApi) throw new Error("assert: tangram api not provided");

// That's the current state of the aircraft
const activeEntity = computed(
  () => tangramApi.state.activeEntity.value as Entity<AircraftState> | null
);
const layerDisposable: Ref<Disposable | null> = ref(null);

export interface Runway {
  latlon: [number, number];
  name: string;
}
const tooltip = reactive<{
  x: number;
  y: number;
  object: Runway | null;
}>({ x: 0, y: 0, object: null });

const updateLayer = async () => {
  if (layerDisposable.value) {
    layerDisposable.value.dispose();
    layerDisposable.value = null;
  }

  console.log("Updating alignment layer", activeEntity.value);
  const icao24 = activeEntity.value?.id;
  const lat = activeEntity.value?.state?.latitude;
  const lon = activeEntity.value?.state?.longitude;

  tooltip.object = null;

  if (activeEntity.value?.type === "jet1090_aircraft") {
    const trajectory = await fetch(`/jet1090/data/${icao24}`)
      .then(response => response.json())
      .catch(error => {
        console.error("Error fetching alignment data:", error);
      });
    console.log("Trajectory data:", trajectory);
    // Implementation of the layer update logic goes here
    const aligned = await fetch("/align/airport", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ aircraft: trajectory })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error("Failed to fetch aligned data");
        }
        return response.json();
      })

      .catch(error => {
        console.error("Error fetching aligned data:", error);
      });
    console.log("Aligned data:", aligned);
    if (aligned.status === "found") {
      const latlon = aligned.latlon as [number, number];
      const alignedLayer = new PathLayer({
        id: "aircraft-align-layer",
        data: [
          {
            path: [
              [lon, lat],
              [latlon[1], latlon[0]]
            ]
          }
        ],
        pickable: false,
        widthScale: 1,
        widthMinPixels: 3,
        extensions: [new PathStyleExtension({ dash: true })],
        getDashArray: () => [10, 5],
        getDashGapPickable: false,
        getDashJustified: true,
        getDashOffset: 0,
        getDashAlign: 0,
        getColor: [0, 0, 0, 255] // Black border
      });

      tooltip.object = {
        latlon: latlon,
        name: aligned.runway
      };

      const projected = project(lat, lon) || { x: 0, y: 0 };
      console.log("Projected runway position:", projected, lat, lon);
      if (projected) {
        tooltip.x = projected.x;
        tooltip.y = projected.y;
      }

      const disposable = tangramApi.map.addLayer(alignedLayer);
      layerDisposable.value = disposable;
    }
  }
};

watch(() => activeEntity.value?.id, updateLayer);

watch(
  () => activeEntity.value?.state,
  state => {
    if (state && tooltip.object) {
      const projected = project(state.latitude, state.longitude) || { x: 0, y: 0 };
      tooltip.object.latlon = [state.latitude, state.longitude];
      console.log("Projected runway position:", projected, tooltip.object.latlon);
      if (projected) {
        tooltip.x = projected.x;
        tooltip.y = projected.y;
      }
    }
  }
);
watch(tangramApi.map.bounds, () => {
  if (tooltip.object) {
    const lat = tooltip.object.latlon[0];
    const lon = tooltip.object.latlon[1];
    const projected = project(lat, lon) || { x: 0, y: 0 };
    console.log("Projected runway position:", projected, tooltip.object.latlon);
    if (projected) {
      tooltip.x = projected.x;
      tooltip.y = projected.y;
    }
  }
});

onUnmounted(() => {
  layerDisposable.value?.dispose();
});

function project(
  lat: number | undefined,
  lon: number | undefined
): undefined | { x: number; y: number } {
  if (lon === undefined || lat === undefined) {
    return undefined;
  }
  return tangramApi?.map?.map?.value?.project([lon, lat]);
}

const tooltipPositionStyle = computed((): CSSProperties => {
  if (!tooltip.object) return { display: "none" };
  console.log(
    "Tooltip projected position:",
    tooltip.x,
    tooltip.y,
    tooltip.object?.latlon
  );
  return {
    position: "absolute",
    left: `${tooltip.x + 15}px`,
    top: `${tooltip.y - 10}px`,
    pointerEvents: "none" as CSSProperties["pointerEvents"],
    background: "#ffe60a",
    border: "1px solid #ccc",
    padding: "4px",
    borderRadius: "4px",
    zIndex: 1000,
    fontFamily: "Frutiger, Helvetica, B612",
    fontWeight: "bold",
    fontSize: "14px"
  };
});
</script>
