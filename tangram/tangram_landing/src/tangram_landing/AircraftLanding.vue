<template>
  <div
    v-for="item in visibleTooltips"
    :key="item.id"
    :style="getTooltipStyle(item)"
    class="map-tooltip"
  >
    {{ item.name }}
  </div>
</template>

<script setup lang="ts">
import {
  computed,
  inject,
  onUnmounted,
  reactive,
  watch,
  ref,
  type CSSProperties,
  type Ref,
} from "vue";
import { PathLayer } from "@deck.gl/layers";
import { PathStyleExtension } from "@deck.gl/extensions";
import type { TangramApi, Disposable } from "@open-aviation/tangram-core/api";
import type { Layer } from "@deck.gl/core";

// TODO: figure out how one plugin can share types with another
interface AircraftState {
  latitude?: number;
  longitude?: number;
}

interface AlignmentData {
  runwayName: string;
  runwayLatLon: [number, number];
}

interface TooltipItem {
  id: string;
  name: string;
  x: number;
  y: number;
}

const tangramApi = inject<TangramApi>("tangramApi");
if (!tangramApi) throw new Error("assert: tangram api not provided");

const layerDisposable: Ref<Disposable | null> = ref(null);
const alignments = reactive(new Map<string, AlignmentData | null>());

const activeAircraft = computed(() => {
  const map = new Map<string, AircraftState>();
  if (!tangramApi.state.activeEntities.value) return map;
  for (const [id, entity] of tangramApi.state.activeEntities.value) {
    if (entity.type === "jet1090_aircraft") {
      map.set(id, entity.state as AircraftState);
    }
  }
  return map;
});

watch(
  () => new Set(activeAircraft.value.keys()),
  async (newIds, oldIds) => {
    if (oldIds) {
      for (const id of oldIds) {
        if (!newIds.has(id)) alignments.delete(id);
      }
    }
    for (const id of newIds) {
      if (!alignments.has(id)) fetchAlignment(id);
    }
  },
  { immediate: true },
);

async function fetchAlignment(id: string) {
  try {
    // TODO: figure out inter-plugin communication to avoid redudant fetches
    // and, do not hardcode jet1090 as the only source of data.
    const trajectory = await fetch(`/jet1090/data/${id}`).then((res) =>
      res.json(),
    );
    const response = await fetch("/align/airport", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ aircraft: trajectory }),
    });
    const aligned = await response.json();

    if (aligned.status === "found" && activeAircraft.value.has(id)) {
      alignments.set(id, {
        runwayName: aligned.runway,
        runwayLatLon: aligned.latlon,
      });
    } else {
      alignments.set(id, null);
    }
  } catch (e) {
    if (activeAircraft.value.has(id)) alignments.set(id, null);
  }
}

const layerData = computed(() => {
  const paths: any[] = [];
  for (const [id, state] of activeAircraft.value) {
    const align = alignments.get(id);
    if (align && state.latitude != null && state.longitude != null) {
      paths.push({
        path: [
          [state.longitude, state.latitude],
          [align.runwayLatLon[1], align.runwayLatLon[0]],
        ],
      });
    }
  }
  return paths;
});

watch(
  layerData,
  (paths) => {
    if (layerDisposable.value) layerDisposable.value.dispose();
    if (paths.length > 0) {
      const layer = new PathLayer({
        id: "aircraft-align-layer",
        data: paths,
        pickable: false,
        widthScale: 1,
        widthMinPixels: 3,
        extensions: [new PathStyleExtension({ dash: true })],
        getDashArray: [10, 5],
        getDashGapPickable: false,
        getDashJustified: true,
        getDashOffset: 0,
        getDashAlign: 0,
        getPath: (d) => d.path,
        getColor: [0, 0, 0, 255],
      }) as Layer;
      layerDisposable.value = tangramApi.map.addLayer(layer);
    }
  },
  { immediate: true },
);

const visibleTooltips = computed(() => {
  const items: TooltipItem[] = [];
  if (!tangramApi.map.isReady.value) return items;

  // required to trigger recompute on map move
  tangramApi.map.center.value;
  tangramApi.map.zoom.value;
  tangramApi.map.pitch.value;
  tangramApi.map.bearing.value;

  const mapInstance = tangramApi.map.getMapInstance();

  for (const [id, align] of alignments) {
    if (!align) continue;
    const [lat, lon] = align.runwayLatLon;
    const projected = mapInstance.project([lon, lat]);
    items.push({
      id,
      name: align.runwayName,
      x: projected.x,
      y: projected.y,
    });
  }
  return items;
});

function getTooltipStyle(item: TooltipItem): CSSProperties {
  return {
    position: "absolute",
    left: `${item.x + 15}px`,
    top: `${item.y - 10}px`,
    pointerEvents: "none",
    background: "#ffe60a",
    border: "1px solid #ccc",
    padding: "4px",
    borderRadius: "4px",
    zIndex: 1000,
    fontFamily: "Frutiger, Helvetica, B612, sans-serif",
    fontWeight: "bold",
    fontSize: "14px",
  };
}

onUnmounted(() => {
  layerDisposable.value?.dispose();
});
</script>
