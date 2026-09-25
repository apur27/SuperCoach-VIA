/** Shown only when JavaScript is off (hidden by the `.js` class set in <head>). */
export function NoScriptNotice({ what, href, linkText }: { what: string; href: string; linkText: string }) {
  return (
    <div className="nojs-only banner banner-info">
      <p>The interactive {what} needs JavaScript. You can still <a href={href}>{linkText}</a>.</p>
    </div>
  );
}
