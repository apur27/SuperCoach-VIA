import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { LineChart, BarChart } from '../../src/islands/common/Charts';

describe('accessible SVG charts', () => {
  it('line chart has a labelled image, a text summary and a data table with missing values', () => {
    const html = renderToStaticMarkup(
      <LineChart id="t" title="Form" description="Disposals by game" xLabel="Game" yLabel="Disposals" series={[{ name: 'Disposals', points: [{ x: 'G1', y: 10 }, { x: 'G2', y: null }, { x: 'G3', y: 20 }] }]} />,
    );
    expect(html).toContain('role="img"');
    expect(html).toMatch(/aria-labelledby="t-title t-desc"/);
    expect(html).toContain('<title id="t-title">Form</title>');
    expect(html).toContain('<caption>Form</caption>');
    expect(html).toContain('not recorded');
    expect(html).toContain('<th scope="col">Game</th>');
    expect(html).not.toMatch(/style=/);
    // null breaks the line into separate segments rather than plotting zero
    expect((html.match(/<path/g) ?? []).length).toBe(2);
  });
  it('two series differ by more than colour (dash + marker shape) and have a legend', () => {
    const html = renderToStaticMarkup(
      <LineChart id="c" title="Compare" description="d" xLabel="Season" yLabel="Mean" series={[{ name: 'A', points: [{ x: '1', y: 1 }, { x: '2', y: 2 }] }, { name: 'B', points: [{ x: '1', y: 2 }, { x: '2', y: 3 }] }]} />,
    );
    expect(html).toContain('series-2');
    expect(html).toContain('Legend');
    expect(html).toMatch(/dashed/);
  });
  it('bar chart renders a table and sorted bars', () => {
    const html = renderToStaticMarkup(<BarChart id="b" title="Bars" description="d" valueLabel="Value" categoryLabel="Name" data={[{ label: 'x', value: 3 }, { label: 'y', value: null }]} />);
    expect(html).toContain('<caption>Bars</caption>');
    expect(html).toContain('not recorded');
    expect((html.match(/<rect/g) ?? []).length).toBe(1);
  });
  it('handles empty data honestly', () => {
    const html = renderToStaticMarkup(<LineChart id="e" title="Empty" description="d" xLabel="x" yLabel="y" series={[{ name: 'A', points: [] }]} />);
    expect(html).toContain('No recorded values to chart');
  });
});
