import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ObjectPerClass } from "../../types/dashboard";

type ObjectsPerClassChartProps = {
  data: ObjectPerClass[];
};

export function ObjectsPerClassChart({ data }: ObjectsPerClassChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-2xl border border-border bg-surface text-sm text-ink-faint">
        Aún no hay categorías.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-card">
      <h3 className="text-sm font-medium text-ink">Objetos por categoría</h3>
      <div className="mt-4 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E7E5E1" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 12, fill: "#8A8782" }}
              axisLine={{ stroke: "#E7E5E1" }}
              tickLine={false}
              interval={0}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fontSize: 12, fill: "#8A8782" }}
              axisLine={false}
              tickLine={false}
              width={32}
            />
            <Tooltip
              cursor={{ fill: "#F5F4F1" }}
              contentStyle={{ borderRadius: 12, border: "1px solid #E7E5E1", fontSize: 12 }}
            />
            <Bar dataKey="count" radius={[6, 6, 0, 0]}>
              {data.map((entry) => (
                // Always the category's own color from the API — never a
                // random/generated palette.
                <Cell key={entry.categoryId} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
