import type { ImageSearchItem } from "@/api/schemas";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ImageCard } from "./ImageCard";

interface ResultsGridProps {
  images: ImageSearchItem[];
  isLoading: boolean;
  error: string | null;
  hasActiveFilters: boolean;
  onRetry: () => void;
  onImageClick: (index: number) => void;
  selectedIds: Set<number>;
  onToggleSelect: (id: number) => void;
}

const SKELETON_COUNT = 10;

export function ResultsGrid({
  images,
  isLoading,
  error,
  hasActiveFilters,
  onRetry,
  onImageClick,
  selectedIds,
  onToggleSelect,
}: ResultsGridProps): JSX.Element {
  if (error) {
    return <ErrorState message={error} onRetry={onRetry} />;
  }

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
        {Array.from({ length: SKELETON_COUNT }).map((_, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: lista fija de placeholders sin identidad propia.
          <Skeleton key={i} className="aspect-square w-full" />
        ))}
      </div>
    );
  }

  if (images.length === 0) {
    return <EmptyState hasFilters={hasActiveFilters} />;
  }

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      {images.map((image, index) => (
        <ImageCard
          key={image.id}
          image={image}
          isSelected={selectedIds.has(image.id)}
          onClick={() => onImageClick(index)}
          onToggleSelect={() => onToggleSelect(image.id)}
        />
      ))}
    </div>
  );
}
