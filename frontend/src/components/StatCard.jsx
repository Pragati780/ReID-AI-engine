import Card from "./Card";

/**
 * src/components/StatCard.jsx
 *
 * A single headline statistic (e.g. "Similarity Score: 0.91") shown at the
 * top of the Results page.
 */
export default function StatCard({ icon: Icon, label, value, accent = "brand" }) {
  const accentClasses = {
    brand: "bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400",
    green: "bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400",
    amber: "bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-400",
    rose: "bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400",
  };

  return (
    <Card className="flex items-center gap-4">
      <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${accentClasses[accent]}`}>
        {Icon && <Icon size={20} strokeWidth={2.2} />}
      </div>
      <div className="min-w-0">
        <p className="truncate text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {label}
        </p>
        <p className="truncate text-xl font-semibold text-slate-900 dark:text-white">{value}</p>
      </div>
    </Card>
  );
}
