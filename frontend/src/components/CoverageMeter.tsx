import { pct } from "../lib/intake";

/**
 * Progress measured in the model's own attribution mass, not in fields filled.
 *
 * "12 of 24 fields" implies every field is worth the same. Summing each field's share of
 * total mean |SHAP| says how much of what the model actually uses is present — so for
 * kidney, supplying urine specific gravity (16.4% of the mass) moves this bar eighty times
 * further than supplying urine sugar (0.2%).
 */
export function CoverageMeter({
  covered,
  coreDone,
  coreTotal,
}: {
  covered: number;
  coreDone: number;
  coreTotal: number;
}) {
  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium">Information the model can use</span>
        <span className="tnum text-sm font-semibold">{pct(covered)}</span>
      </div>
      <div
        className="mt-2 h-2 overflow-hidden rounded-full bg-canvas"
        role="progressbar"
        aria-valuenow={Math.round(covered * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Share of the model's attribution mass supplied"
      >
        <div
          className="h-full rounded-full bg-accent transition-all duration-300"
          style={{ width: `${Math.min(100, covered * 100)}%` }}
        />
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted">
        Weighted by how much each field actually contributes to this model, not by how many
        boxes are filled.{" "}
        <span className="tnum font-medium text-ink">
          {coreDone} of {coreTotal}
        </span>{" "}
        key fields provided.
      </p>
    </div>
  );
}
