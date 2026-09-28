import { type FormEvent, useState } from "react";

interface SearchBarProps {
  initialValue: string;
  onSearch: (q: string) => void;
}

export function SearchBar({ initialValue, onSearch }: SearchBarProps): JSX.Element {
  const [value, setValue] = useState(initialValue);

  function handleSubmit(e: FormEvent): void {
    e.preventDefault();
    onSearch(value);
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2">
      <div className="relative flex-1">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder='Busca por texto libre, ej: "car AND person"'
          className="w-full rounded-xl border border-border-strong bg-surface px-4 py-2.5 text-sm text-ink placeholder:text-ink-faint focus:border-accent-lilac"
        />
      </div>
      <button
        type="submit"
        className="shrink-0 rounded-xl bg-ink px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-ink/90"
      >
        Buscar
      </button>
    </form>
  );
}
