/** Small accessible SVG charts with table equivalents. No inline styles (CSP). */
import { formatStat } from '../../lib/format';

export interface Point { x: string; y: number | null }
export interface Series { name: string; points: Point[] }

const W = 640;
const H = 260;
const PAD = { l: 48, r: 16, t: 16, b: 40 };

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const mag = 10 ** Math.floor(Math.log10(v));
  return Math.ceil(v / mag) * mag;
}

function DataTable({ title, xLabel, series }: { title: string; xLabel: string; series: Series[] }) {
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))];
  return (
    <details className="data-table">
      <summary>Data table: {title}</summary>
      <div className="table-wrap">
        <table>
          <caption>{title}</caption>
          <thead>
            <tr>
              <th scope="col">{xLabel}</th>
              {series.map((s) => <th scope="col" className="num" key={s.name}>{s.name}</th>)}
            </tr>
          </thead>
          <tbody>
            {xs.map((x) => (
              <tr key={x}>
                <th scope="row">{x}</th>
                {series.map((s) => {
                  const y = s.points.find((p) => p.x === x)?.y ?? null;
                  return <td className="num" key={s.name}>{y === null ? <span className="missing">not recorded</span> : formatStat(y, 1)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function LineChart({ id, title, description, xLabel, yLabel, series }: { id: string; title: string; description: string; xLabel: string; yLabel: string; series: Series[] }) {
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))];
  const values = series.flatMap((s) => s.points.map((p) => p.y)).filter((v): v is number => v !== null);
  if (values.length === 0) {
    return (
      <figure className="chart">
        <figcaption><strong>{title}</strong></figcaption>
        <p className="missing">No recorded values to chart.</p>
      </figure>
    );
  }
  const max = niceMax(Math.max(...values));
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const xPos = (i: number) => PAD.l + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const yPos = (v: number) => PAD.t + ih - (v / max) * ih;
  const ticks = [0, max / 2, max];
  const stride = Math.max(1, Math.ceil(xs.length / 10));
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>{title}</title>
        <desc id={`${id}-desc`}>{description}. Missing values are gaps, not zero. Full values are in the data table below.</desc>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={PAD.l} x2={W - PAD.r} y1={yPos(t)} y2={yPos(t)} />
            <text x={PAD.l - 6} y={yPos(t) + 4} textAnchor="end">{formatStat(t, t % 1 ? 1 : 0)}</text>
          </g>
        ))}
        <line className="axis" x1={PAD.l} x2={W - PAD.r} y1={PAD.t + ih} y2={PAD.t + ih} />
        {xs.map((x, i) => (i % stride === 0 ? <text key={x} x={xPos(i)} y={H - PAD.b + 16} textAnchor="middle">{x}</text> : null))}
        <text x={PAD.l + iw / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
        <text x={12} y={PAD.t + ih / 2} textAnchor="middle" transform={`rotate(-90 12 ${PAD.t + ih / 2})`}>{yLabel}</text>
        {series.map((s, si) => {
          const cls = si === 0 ? '1' : '2';
          const segments: string[] = [];
          let current: string[] = [];
          xs.forEach((x, i) => {
            const y = s.points.find((p) => p.x === x)?.y ?? null;
            if (y === null) {
              if (current.length) segments.push(current.join(' '));
              current = [];
            } else current.push(`${current.length ? 'L' : 'M'}${xPos(i).toFixed(1)},${yPos(y).toFixed(1)}`);
          });
          if (current.length) segments.push(current.join(' '));
          return (
            <g key={s.name}>
              {segments.map((d) => <path key={d} d={d} className={`series-${cls}`} />)}
              {xs.map((x, i) => {
                const y = s.points.find((p) => p.x === x)?.y ?? null;
                if (y === null) return null;
                return si === 0
                  ? <circle key={x} cx={xPos(i)} cy={yPos(y)} r={4} className="point-1" />
                  : <rect key={x} x={xPos(i) - 4} y={yPos(y) - 4} width={8} height={8} className="point-2" />;
              })}
            </g>
          );
        })}
      </svg>
      <figcaption>
        <strong>{title}</strong>
        {series.length > 1 ? (
          <span> — Legend: {series.map((s, i) => `${s.name} (${i === 0 ? 'solid line, round markers' : 'dashed line, square markers'})`).join('; ')}</span>
        ) : null}
      </figcaption>
      <DataTable title={title} xLabel={xLabel} series={series} />
    </figure>
  );
}

export function BarChart({ id, title, description, categoryLabel, valueLabel, data }: { id: string; title: string; description: string; categoryLabel: string; valueLabel: string; data: { label: string; value: number | null }[] }) {
  const present = data.filter((d): d is { label: string; value: number } => d.value !== null);
  const max = niceMax(Math.max(0, ...present.map((d) => d.value)));
  const rowH = 26;
  const labelW = 170;
  const h = PAD.t + rowH * Math.max(1, data.length) + 24;
  const iw = W - labelW - PAD.r - 40;
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>{title}</title>
        <desc id={`${id}-desc`}>{description}. Bars without a value are not recorded, not zero.</desc>
        {data.map((d, i) => {
          const y = PAD.t + i * rowH;
          return (
            <g key={`${d.label}-${i}`}>
              <text x={labelW - 6} y={y + 17} textAnchor="end">{d.label.length > 26 ? `${d.label.slice(0, 25)}…` : d.label}</text>
              {d.value === null ? (
                <text x={labelW + 4} y={y + 17}>not recorded</text>
              ) : (
                <>
                  <rect x={labelW} y={y + 4} width={Math.max(1, (d.value / max) * iw)} height={rowH - 8} className="bar-1" />
                  <text x={labelW + Math.max(1, (d.value / max) * iw) + 4} y={y + 17}>{formatStat(d.value, d.value % 1 ? 1 : 0)}</text>
                </>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption><strong>{title}</strong></figcaption>
      <DataTable title={title} xLabel={categoryLabel} series={[{ name: valueLabel, points: data.map((d) => ({ x: d.label, y: d.value })) }]} />
    </figure>
  );
}
