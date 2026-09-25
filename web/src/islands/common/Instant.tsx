import { useEffect, useState } from 'react';
import { formatInstant, zoneLabel, type ZoneChoice } from '../../lib/format';
import { getZone, ZONE_EVENT } from '../../lib/prefs';

export function useZone(): ZoneChoice {
  const [zone, setZone] = useState<ZoneChoice>('Australia/Melbourne');
  useEffect(() => {
    setZone(getZone());
    const on = () => setZone(getZone());
    window.addEventListener(ZONE_EVENT, on);
    return () => window.removeEventListener(ZONE_EVENT, on);
  }, []);
  return zone;
}

export function Instant({ iso }: { iso: string | null | undefined }) {
  const zone = useZone();
  if (!iso) return <span className="missing">not recorded</span>;
  return <time dateTime={iso} title={zoneLabel(zone)}>{formatInstant(iso, zone)}</time>;
}
