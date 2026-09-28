import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useCategories } from "../../hooks/useCategories";
import { useImageAnnotations } from "../../hooks/useImageAnnotations";
import { useToasts } from "../../hooks/useToasts";
import { getImageFileUrl } from "../../lib/api/images";
import type { AnnotateNavigationState } from "../../types/navigation";
import { ToastStack } from "../shared/ToastStack";
import { Stepper } from "../ui/Stepper";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { AnnotationsSidebar } from "./AnnotationsSidebar";
import { LeaveConfirmModal } from "./LeaveConfirmModal";
import { QueueNav } from "./QueueNav";
import { Toolbar } from "./Toolbar";

const VIEWPORT_PADDING_PX = 48;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4;
const SEARCH_SCREEN_PATH = "/search";
const UPLOAD_SCREEN_PATH = "/upload";
const UPLOAD_STEPS = ["Subir fotografías", "Identificar objetos"];

function parseQueue(param: string | null, currentId: number): number[] {
  if (!param) return [];
  const ids = param
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  return ids.includes(currentId) ? ids : ids.length > 0 ? [currentId, ...ids] : [];
}

export function AnnotateScreen() {
  const params = useParams<{ imageId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();

  // Estado opcional pasado por Upload/Search: ver types/navigation.ts. Nunca
  // es requerido — si viene vacío, se usan valores por defecto.
  const navState = (location.state as AnnotateNavigationState | null) ?? null;

  // "Volver" regresa a la pantalla desde donde se entró (Upload o Search);
  // sin ese dato (p. ej. URL directa) se usa Search como default.
  const backTarget = navState?.from === "upload" ? UPLOAD_SCREEN_PATH : SEARCH_SCREEN_PATH;
  const backLabel = navState?.from === "upload" ? "Subir fotografías" : "Buscar";

  const imageId = Number(params.imageId);
  const queue = useMemo(
    () => parseQueue(searchParams.get("queue"), imageId),
    [searchParams, imageId]
  );
  const currentIndex = queue.indexOf(imageId);

  const [zoomMultiplier, setZoomMultiplier] = useState(1);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [queueDone, setQueueDone] = useState(false);
  const [showLeaveModal, setShowLeaveModal] = useState(false);
  const [isSavingDraft, setIsSavingDraft] = useState(false);

  // Tamaño real disponible para dibujar la imagen, medido del contenedor.
  // Se recalcula si la ventana cambia de tamaño, para que la imagen siempre
  // quepa completa en pantalla (sin scroll de página) al 100%.
  const viewportRef = useRef<HTMLDivElement>(null);
  const [viewportSize, setViewportSize] = useState({ width: 900, height: 600 });

  const { toasts, showToast, dismissToast } = useToasts();
  const {
    categories,
    isLoading: categoriesLoading,
    error: categoriesError,
    retry: retryCategories,
  } = useCategories();
  const {
    annotations,
    imageMeta,
    isLoading,
    loadError,
    reload,
    selectedId,
    setSelectedId,
    createBox,
    commitBoxChange,
    removeBox,
    finalize,
    saveDraft,
    undo,
    canUndo,
  } = useImageAnnotations(imageId, showToast, {
    initialFilename: navState?.filenames?.[imageId],
    initialStatus: navState?.statuses?.[imageId],
  });

  // biome-ignore lint/correctness/useExhaustiveDependencies: vuelve a medir cuando cambia el estado de carga/imagen, aunque el efecto solo lea el ref.
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const measure = () => setViewportSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [isLoading, loadError, imageMeta]);

  const baseFitScale = imageMeta
    ? Math.max(
        0.05,
        Math.min(
          1,
          (viewportSize.width - VIEWPORT_PADDING_PX) / imageMeta.width,
          (viewportSize.height - VIEWPORT_PADDING_PX) / imageMeta.height
        )
      )
    : 1;
  const displayScale = baseFitScale * zoomMultiplier;

  const handleZoomIn = useCallback(
    () => setZoomMultiplier((z) => Math.min(MAX_ZOOM, +(z * 1.2).toFixed(2))),
    []
  );
  const handleZoomOut = useCallback(
    () => setZoomMultiplier((z) => Math.max(MIN_ZOOM, +(z / 1.2).toFixed(2))),
    []
  );
  const handleZoomReset = useCallback(() => setZoomMultiplier(1), []);
  const handleWheelZoom = useCallback((deltaY: number) => {
    setZoomMultiplier((z) => {
      const next = deltaY > 0 ? z / 1.08 : z * 1.08;
      return +Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next)).toFixed(2);
    });
  }, []);

  const handleUndoKey = useCallback(
    (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        void undo();
      }
    },
    [undo]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleUndoKey);
    return () => window.removeEventListener("keydown", handleUndoKey);
  }, [handleUndoKey]);

  const goToQueueIndex = useCallback(
    (index: number) => {
      const targetId = queue[index];
      if (targetId === undefined) return;
      navigate(`/annotate/${targetId}?queue=${queue.join(",")}`, { state: navState ?? undefined });
    },
    [navigate, navState, queue]
  );

  const handleFinalize = useCallback(async () => {
    setIsFinalizing(true);
    const ok = await finalize();
    setIsFinalizing(false);
    // Al finalizar, se cierra la pantalla y se regresa a donde se entró
    // (Search/Upload) en vez de dejar al usuario viendo la imagen ya
    // completada.
    if (ok) navigate(backTarget);
  }, [finalize, navigate, backTarget]);

  const handleFinishQueue = useCallback(() => {
    setQueueDone(true);
  }, []);

  // Si ya hay objetos dibujados y la imagen no está completada, el status
  // "in_progress" solo vive en el estado local (ver comentario de
  // `saveDraft`) — antes de salir, se le da al usuario la opción explícita
  // de persistirlo como borrador o de salir sin guardarlo.
  const handleBackClick = useCallback(() => {
    const hasUnsavedProgress = annotations.length > 0 && imageMeta?.status !== "completed";
    if (hasUnsavedProgress) {
      setShowLeaveModal(true);
    } else {
      navigate(backTarget);
    }
  }, [annotations.length, imageMeta?.status, navigate, backTarget]);

  const handleDiscardAndLeave = useCallback(() => {
    setShowLeaveModal(false);
    navigate(backTarget);
  }, [navigate, backTarget]);

  const handleSaveDraftAndLeave = useCallback(async () => {
    setIsSavingDraft(true);
    const saved = await saveDraft();
    setIsSavingDraft(false);
    if (saved) {
      setShowLeaveModal(false);
      navigate(backTarget);
    }
  }, [navigate, saveDraft, backTarget]);

  if (Number.isNaN(imageId)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 px-6 text-center text-sm text-ink-muted">
        <p>La URL no incluye un ID de imagen válido.</p>
        <button
          type="button"
          onClick={() => navigate(backTarget)}
          className="rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-sidebar"
        >
          ← Volver a {backLabel}
        </button>
      </div>
    );
  }

  if (queueDone) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 px-6 text-center">
        <h1 className="text-lg font-semibold text-ink">Cola completada</h1>
        <p className="max-w-lg text-sm text-ink-muted">
          Anotaste todas las imágenes seleccionadas. Puedes volver para elegir más.
        </p>
        <button
          type="button"
          onClick={() => navigate(backTarget)}
          className="rounded-xl bg-accent-lilac px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-accent-lilac/90"
        >
          Volver a {backLabel}
        </button>
      </div>
    );
  }

  const showToolbar = !isLoading && !loadError && imageMeta;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas">
      {showToolbar ? (
        <Toolbar
          filename={imageMeta.filename}
          status={imageMeta.status}
          zoom={displayScale / baseFitScale}
          onBack={handleBackClick}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onZoomReset={handleZoomReset}
          canUndo={canUndo}
          onUndo={undo}
          onFinalize={handleFinalize}
          isFinalizing={isFinalizing}
        />
      ) : (
        <div className="flex items-center gap-3 border-b border-border bg-surface px-4 py-2.5">
          <button
            type="button"
            onClick={handleBackClick}
            className="rounded-lg px-2 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-sidebar hover:text-ink"
          >
            ← Volver
          </button>
        </div>
      )}

      {navState?.from === "upload" && (
        <div className="border-b border-border bg-surface px-4 py-2">
          <Stepper steps={UPLOAD_STEPS} currentIndex={1} />
        </div>
      )}

      {queue.length > 0 && (
        <div className="border-b border-border bg-surface px-4 py-2">
          <QueueNav
            queue={queue}
            currentIndex={Math.max(currentIndex, 0)}
            onGoTo={goToQueueIndex}
            onFinishQueue={handleFinishQueue}
          />
        </div>
      )}

      {isLoading && (
        <div className="flex flex-1 items-center justify-center text-sm text-ink-faint">
          Cargando imagen y anotaciones…
        </div>
      )}

      {!isLoading && loadError && (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 text-center">
          <p className="text-sm text-red-700">{loadError}</p>
          <button
            type="button"
            onClick={reload}
            className="rounded-lg border border-red-200 bg-surface px-3 py-1.5 text-sm font-medium text-red-700 transition-colors hover:bg-red-50"
          >
            Reintentar
          </button>
        </div>
      )}

      {!isLoading && !loadError && imageMeta && (
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <main
            ref={viewportRef}
            className="relative flex min-h-[45vh] flex-1 items-center justify-center overflow-auto p-4 lg:min-h-0 lg:p-6"
          >
            {categoriesError ? (
              <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                <span className="flex-1">{categoriesError}</span>
                <button
                  type="button"
                  onClick={retryCategories}
                  className="rounded-lg border border-red-200 bg-surface px-2 py-1 text-xs font-medium transition-colors hover:bg-red-100"
                >
                  Reintentar
                </button>
              </div>
            ) : (
              <AnnotationCanvas
                imageUrl={getImageFileUrl(imageId)}
                imageWidth={imageMeta.width}
                imageHeight={imageMeta.height}
                annotations={annotations}
                categories={categoriesLoading ? [] : categories}
                selectedId={selectedId}
                onSelect={setSelectedId}
                displayScale={displayScale}
                onWheelZoom={handleWheelZoom}
                onCreateBox={createBox}
                onCommitChange={commitBoxChange}
                onDelete={removeBox}
              />
            )}
          </main>

          <AnnotationsSidebar
            annotations={annotations}
            imageUrl={getImageFileUrl(imageId)}
            imageWidth={imageMeta.width}
            imageHeight={imageMeta.height}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onDelete={(id) => void removeBox(id)}
          />
        </div>
      )}

      <ToastStack toasts={toasts} onDismiss={dismissToast} />

      {showLeaveModal && (
        <LeaveConfirmModal
          objectCount={annotations.length}
          isSaving={isSavingDraft}
          onStay={() => setShowLeaveModal(false)}
          onDiscard={handleDiscardAndLeave}
          onSaveDraft={() => void handleSaveDraftAndLeave()}
        />
      )}
    </div>
  );
}
