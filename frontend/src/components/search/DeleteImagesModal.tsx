import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";

interface DeleteImagesModalProps {
  count: number;
  isDeleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export function DeleteImagesModal({
  count,
  isDeleting,
  onCancel,
  onConfirm,
}: DeleteImagesModalProps): JSX.Element {
  return (
    <Modal
      title={`¿Eliminar ${count} ${count === 1 ? "foto" : "fotos"}?`}
      description="Esta acción no se puede deshacer: se borran las imágenes, sus anotaciones y sus archivos."
      onClose={onCancel}
    >
      <div className="flex justify-end gap-2">
        <Button variant="ghost" disabled={isDeleting} onClick={onCancel}>
          Cancelar
        </Button>
        <Button variant="destructive" isLoading={isDeleting} onClick={onConfirm}>
          {isDeleting ? "Eliminando…" : "Eliminar"}
        </Button>
      </div>
    </Modal>
  );
}
