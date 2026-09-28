import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";

interface LeaveConfirmModalProps {
  objectCount: number;
  isSaving: boolean;
  onStay: () => void;
  onDiscard: () => void;
  onSaveDraft: () => void;
}

export function LeaveConfirmModal({
  objectCount,
  isSaving,
  onStay,
  onDiscard,
  onSaveDraft,
}: LeaveConfirmModalProps) {
  return (
    <Modal
      title="¿Salir de la anotación?"
      onClose={onStay}
      description={
        <>
          Esta imagen todavía no está marcada como completada
          {objectCount > 0 && (
            <>
              , pero ya tiene {objectCount} objeto{objectCount === 1 ? "" : "s"} anotado
              {objectCount === 1 ? "" : "s"}
            </>
          )}
          . Puedes guardarla como borrador ("En progreso") para continuar después.
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <Button variant="primary" isLoading={isSaving} onClick={onSaveDraft} className="w-full">
          {isSaving ? "Guardando…" : "Guardar borrador y salir"}
        </Button>
        <Button variant="secondary" disabled={isSaving} onClick={onDiscard} className="w-full">
          Salir sin guardar cambios
        </Button>
        <Button variant="ghost" onClick={onStay} className="w-full">
          Seguir editando
        </Button>
      </div>
    </Modal>
  );
}
