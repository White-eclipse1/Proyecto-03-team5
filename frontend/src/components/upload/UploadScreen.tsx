import { useNavigate } from "react-router-dom";
import { useToasts } from "../../hooks/useToasts";
import { useUploadQueue } from "../../hooks/useUploadQueue";
import type { AnnotateNavigationState } from "../../types/navigation";
import type { ImageStatus } from "../../types/schemas";
import { ToastStack } from "../shared/ToastStack";
import { Button } from "../ui/Button";
import { Stepper } from "../ui/Stepper";
import { Dropzone } from "./Dropzone";
import { PendingImageList } from "./PendingImageList";
import { UploadFeedbackList } from "./UploadFeedbackList";

const UPLOAD_STEPS = ["Subir fotografías", "Identificar objetos"];

export function UploadScreen() {
  const navigate = useNavigate();
  const { toasts, showToast, dismissToast } = useToasts();
  const {
    uploadItems,
    pendingEntries,
    isLoadingPending,
    pendingError,
    selectedIds,
    addFiles,
    retryUpload,
    toggleSelect,
    deleteEntry,
    reloadPending,
  } = useUploadQueue(showToast);

  const selectedList = Array.from(selectedIds);

  const handleAnnotateSelected = () => {
    if (selectedList.length === 0) return;

    // Esta lista incluye tanto 'pending' como 'in_progress' (ver
    // searchPendingImages) — se pasa el status real de cada entrada por
    // estado de navegación para que el Canvas no tenga que adivinarlo ni
    // depender de un endpoint adicional.
    const filenames: Record<number, string> = {};
    const statuses: Record<number, ImageStatus> = {};
    for (const entry of pendingEntries) {
      if (selectedIds.has(entry.id)) {
        filenames[entry.id] = entry.filename;
        statuses[entry.id] = entry.status;
      }
    }
    const state: AnnotateNavigationState = { filenames, statuses, from: "upload" };

    const [first, ...rest] = selectedList;
    if (rest.length === 0) {
      navigate(`/annotate/${first}`, { state });
    } else {
      navigate(`/annotate/${first}?queue=${selectedList.join(",")}`, { state });
    }
  };

  return (
    <main className="flex-1 px-6 py-8 lg:px-10">
      <div className="mx-auto flex max-w-5xl flex-col gap-8">
        <header className="flex flex-col gap-3">
          <Stepper steps={UPLOAD_STEPS} currentIndex={0} />
          <div>
            <h1 className="text-xl font-semibold text-ink">Subir fotografías</h1>
            <p className="mt-1 text-sm text-ink-muted">
              Arrastra fotografías o selecciónalas desde tu equipo. Cuando termines, elige cuáles
              quieres identificar y pasa al siguiente paso.
            </p>
          </div>
        </header>

        <Dropzone onFiles={addFiles} />

        <UploadFeedbackList items={uploadItems} onRetry={retryUpload} />

        <section className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-ink">Pendientes por identificar</h2>
            <span className="text-xs text-ink-faint">
              {selectedList.length > 0 ? `${selectedList.length} seleccionada(s)` : ""}
            </span>
          </div>
          <PendingImageList
            entries={pendingEntries}
            isLoading={isLoadingPending}
            error={pendingError}
            selectedIds={selectedIds}
            onToggle={toggleSelect}
            onDelete={(id) => void deleteEntry(id)}
            onRetry={reloadPending}
          />
        </section>

        <div className="sticky bottom-6 flex justify-end">
          <Button
            variant="primary"
            size="md"
            disabled={selectedList.length === 0}
            onClick={handleAnnotateSelected}
          >
            Identificar objetos
            {selectedList.length > 0 ? ` (${selectedList.length})` : ""}
            <span aria-hidden>→</span>
          </Button>
        </div>

        <ToastStack toasts={toasts} onDismiss={dismissToast} />
      </div>
    </main>
  );
}
