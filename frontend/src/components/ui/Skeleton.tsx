interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps): JSX.Element {
  return <div className={`animate-pulse rounded-xl bg-border ${className}`} />;
}
