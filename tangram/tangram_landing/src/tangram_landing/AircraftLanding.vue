<template><div hidden></div></template>

<script setup lang="ts">
import type { Layer } from "@deck.gl/core";
import {
  PathStyleExtension,
  type PathStyleExtensionProps,
} from "@deck.gl/extensions";
import { PathLayer, TextLayer, type PathLayerProps } from "@deck.gl/layers";
import {
  TrajectoryApi,
  type Disposable,
  type EntityKey,
  type TangramApi,
  type TrajectoryGetRequest,
  type TrajectoryGetResult,
} from "@open-aviation/tangram-core/api";
import { computed, inject, onUnmounted, ref, shallowRef, watch } from "vue";

// TODO: hardcoding jet1090 for now, we may want to generalise so it handles say
// minisky
const ENTITY_TYPE = "jet1090_aircraft";
const TRAJECTORY_TIMEOUT_MS = 30_000;
const ROUTE_COLOR = [154, 146, 98, 128] as const;
const LABEL_COLOR = [255, 250, 226, 230] as const;
const LABEL_BACKGROUND_COLOR = [92, 87, 57, 170] as const;
const LABEL_BORDER_COLOR = [154, 146, 98, 112] as const;

type MapPosition = [longitude: number, latitude: number];
type LatitudeLongitude = [latitude: number, longitude: number];

interface AircraftState {
  latitude?: number;
  longitude?: number;
}

interface AlignmentData {
  runwayName: string;
  runwayLatLon: LatitudeLongitude;
}

interface AlignmentFoundResponse {
  status: "found";
  airport: string;
  runway: string;
  latlon: LatitudeLongitude;
}

interface AlignmentNotFoundResponse {
  status: "not found";
}

type AlignmentResponse = AlignmentFoundResponse | AlignmentNotFoundResponse;

interface SelectedAlignment {
  aircraftId: string;
  data: AlignmentData;
}

interface AlignmentPath {
  path: [MapPosition, MapPosition];
}

interface RunwayLabel {
  position: MapPosition;
  text: string;
}

const tangramApi = inject<TangramApi>("tangramApi")!;
const aircraft = tangramApi.state.getEntitiesByType<AircraftState>(ENTITY_TYPE);
const selectedAircraftId = ref<string | null>(null);
const alignment = shallowRef<SelectedAlignment | null>(null);
const routeLayerDisposable = shallowRef<Disposable | null>(null);
const labelLayerDisposable = shallowRef<Disposable | null>(null);

const selectionDisposable = tangramApi.selection.onChanged((selection) => {
  // intentionally follow one aircraft from a multi-selection for now
  selectedAircraftId.value =
    selection.get(ENTITY_TYPE)?.values().next().value ?? null;
});

const selectedAircraft = computed(() => {
  const id = selectedAircraftId.value;
  return id ? (aircraft.value.get(id) ?? null) : null;
});

watch(
  selectedAircraftId,
  (id, _previousId, onCleanup) => {
    alignment.value = null;
    if (!id) return;

    const controller = new AbortController();
    onCleanup(() => controller.abort());
    void loadAlignment(id, controller.signal);
  },
  { immediate: true },
);

async function loadAlignment(id: string, signal: AbortSignal): Promise<void> {
  try {
    const data = await fetchAlignment(id, signal);
    if (!signal.aborted && selectedAircraftId.value === id) {
      alignment.value = data ? { aircraftId: id, data } : null;
    }
  } catch {
    if (!signal.aborted && selectedAircraftId.value === id) {
      alignment.value = null;
    }
  }
}

async function fetchAlignment(
  id: string,
  signal: AbortSignal,
): Promise<AlignmentData | null> {
  const key: EntityKey = { id, type: ENTITY_TYPE };
  const trajectory = await tangramApi.bus.request<
    Omit<TrajectoryGetRequest, "request_id">,
    TrajectoryGetResult
  >(
    TrajectoryApi.TOPIC_GET,
    { key },
    { signal, timeoutMs: TRAJECTORY_TIMEOUT_MS },
  );

  const response = await fetch("/align/airport", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ aircraft: trajectory.points }),
    signal,
  });
  if (!response.ok) {
    throw new Error(`alignment request failed: ${response.status}`);
  }

  const result = (await response.json()) as AlignmentResponse;
  if (result.status === "not found") return null;
  return { runwayName: result.runway, runwayLatLon: result.latlon };
}

const overlayData = computed(() => {
  const currentAlignment = alignment.value;
  const currentAircraft = selectedAircraft.value;
  if (!currentAlignment || !currentAircraft) return null;

  const { latitude, longitude } = currentAircraft.state;
  if (latitude == null || longitude == null) return null;

  const [runwayLatitude, runwayLongitude] = currentAlignment.data.runwayLatLon;
  const runwayPosition: MapPosition = [runwayLongitude, runwayLatitude];

  return {
    paths: [
      {
        path: [[longitude, latitude], runwayPosition],
      },
    ] satisfies AlignmentPath[],
    labels: [
      {
        position: runwayPosition,
        text: currentAlignment.data.runwayName,
      },
    ] satisfies RunwayLabel[],
  };
});

watch(
  overlayData,
  (data) => {
    if (!data) {
      routeLayerDisposable.value?.dispose();
      routeLayerDisposable.value = null;
      labelLayerDisposable.value?.dispose();
      labelLayerDisposable.value = null;
      return;
    }

    const routeProps: PathLayerProps<AlignmentPath> &
      PathStyleExtensionProps<AlignmentPath> = {
      id: "aircraft-align-route",
      data: data.paths,
      pickable: false,
      widthScale: 1,
      widthMinPixels: 2,
      getWidth: 2,
      getPath: (item) => item.path,
      getColor: ROUTE_COLOR,
      extensions: [new PathStyleExtension({ dash: true })],
      getDashArray: [10, 10],
      dashJustified: true,
    };
    const routeLayer = new PathLayer<AlignmentPath>(routeProps) as Layer;

    const labelLayer = new TextLayer<RunwayLabel>({
      id: "aircraft-align-runway-label",
      data: data.labels,
      pickable: false,
      billboard: true,
      background: true,
      backgroundPadding: [4, 2],
      backgroundBorderRadius: 3,
      fontFamily: "B612",
      fontWeight: 600,
      getPosition: (item) => item.position,
      getText: (item) => item.text,
      getSize: 12,
      getColor: LABEL_COLOR,
      getBackgroundColor: LABEL_BACKGROUND_COLOR,
      getBorderColor: LABEL_BORDER_COLOR,
      getBorderWidth: 1,
      getPixelOffset: [0, 12],
    }) as Layer;

    if (routeLayerDisposable.value) {
      tangramApi.map.setLayer(routeLayer, { slot: "routes" });
    } else {
      routeLayerDisposable.value = tangramApi.map.setLayer(routeLayer, {
        slot: "routes",
      });
    }

    if (labelLayerDisposable.value) {
      tangramApi.map.setLayer(labelLayer, { slot: "routes" });
    } else {
      labelLayerDisposable.value = tangramApi.map.setLayer(labelLayer, {
        slot: "routes",
      });
    }
  },
  { immediate: true },
);

onUnmounted(() => {
  routeLayerDisposable.value?.dispose();
  labelLayerDisposable.value?.dispose();
  selectionDisposable.dispose();
});
</script>
