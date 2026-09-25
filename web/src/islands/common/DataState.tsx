import type { ReactNode } from 'react';
import type { LoadState } from './runtime';

interface Props<T> {
  state: LoadState<T>;
  retry: () => void;
  what: string;
  children: (data: T, opts: { stale: boolean }) => ReactNode;
  idle?: ReactNode;
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  notFound?: ReactNode;
}

/** Loading / empty / error+retry / stale (revalidating) / success for one resource. */
export function DataState<T>({ state, retry, what, children, idle, isEmpty, empty, notFound }: Props<T>) {
  if (state.status === 'idle') return <>{idle ?? null}</>;
  if (state.status === 'loading') {
    if (state.previous !== undefined) {
      return (
        <div aria-busy="true">
          <p className="banner banner-info" role="status">Updating {what}… showing the previous view until it loads.</p>
          {children(state.previous, { stale: true })}
        </div>
      );
    }
    return <p role="status" className="muted" data-state="loading">Loading {what}…</p>;
  }
  if (state.status === 'error' && state.kind === 'notfound') {
    return <>{notFound ?? <div className="banner banner-info" data-state="notfound"><p><strong>Not found.</strong> This {what} is not part of the current release.</p></div>}</>;
  }
  if (state.status === 'error') {
    const title =
      state.kind === 'release'
        ? 'A newer release is available.'
        : state.kind === 'invalid'
          ? `The ${what} data could not be verified, so it is not shown.`
          : `Could not load ${what}.`;
    return (
      <div>
        <div className="banner banner-error" role="alert" data-state="error" data-error-kind={state.kind}>
          <p><strong>{title}</strong></p>
          {state.kind === 'release' ? (
            <p>This page belongs to an older release whose data is no longer served. Reload to switch to the newest release; data from different releases is never mixed.</p>
          ) : state.kind === 'invalid' ? (
            <p>The file did not match the published data contract. Reloading may pick up a newer, valid release.</p>
          ) : (
            <p>Check your connection and try again.</p>
          )}
          <p className="cluster">
            {state.kind === 'release' || state.kind === 'invalid' ? (
              <button type="button" onClick={() => window.location.reload()}>Reload page</button>
            ) : null}
            {state.kind !== 'release' ? <button type="button" className="secondary" onClick={retry}>Try again</button> : null}
          </p>
        </div>
        {state.previous !== undefined ? (
          <>
            <p className="banner banner-stale">Showing the last successfully loaded {what}; it may be out of date.</p>
            {children(state.previous, { stale: true })}
          </>
        ) : null}
      </div>
    );
  }
  if (isEmpty?.(state.data)) return <>{empty ?? <p data-state="empty">No {what} in this release.</p>}</>;
  return <>{children(state.data, { stale: false })}</>;
}
