import { useRef, useState } from "react";

/**
 * The upload surface.
 *
 * It advertises PDF only, because that is what extraction currently handles. Listing JPG
 * and PNG would set an expectation the server then refuses with a 415 — an affordance that
 * does not work is worse than one that is absent, so images are named as *not yet* supported
 * and get a specific message if dropped rather than a generic failure.
 */
export function UploadDropzone({
  onFile,
  busy,
  disabled,
}: {
  onFile: (file: File) => void;
  busy?: boolean;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handle = (file: File | undefined) => {
    setLocalError(null);
    if (!file) return;
    const isPdf =
      file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) {
      setLocalError(
        "That looks like an image. Reading photographed or scanned documents isn't " +
          "available yet — upload a PDF, or enter your information manually below.",
      );
      return;
    }
    onFile(file);
  };

  const interactive = !busy && !disabled;

  return (
    <div>
      <div
        role="button"
        tabIndex={interactive ? 0 : -1}
        aria-disabled={!interactive}
        aria-label="Upload your report"
        onClick={() => interactive && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (interactive && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (interactive) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (interactive) handle(e.dataTransfer.files?.[0]);
        }}
        // `.surface` inverts the ground against the page in both themes and re-points every
        // token inside it; `.dropzone` softens that ground to charcoal in light mode.
        className={`surface dropzone flex flex-col items-center justify-center rounded-2xl
          border-2 border-dashed px-6 py-16 text-center transition-all duration-200
          ${over ? "border-accent" : "border-accent-line hover:border-accent"}
          ${interactive ? "cursor-pointer" : "cursor-default opacity-70"}`}
      >
        {busy ? (
          <>
            <span className="h-7 w-7 animate-spin rounded-full border-2 border-line border-t-accent" />
            <p className="mt-5 text-base font-medium text-ink">Reading your report…</p>
            <p className="mt-1.5 text-sm text-muted">This takes a moment.</p>
          </>
        ) : (
          <>
            <span aria-hidden className="text-3xl">
              📄
            </span>
            <p className="mt-5 text-lg font-semibold text-ink">Upload your report</p>
            <p className="mt-1.5 text-sm text-muted">PDF</p>
            <p className="mt-5 text-sm text-body">
              Drag &amp; drop, or{" "}
              <span className="font-medium text-accent underline underline-offset-4">
                browse
              </span>
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={(e) => handle(e.target.files?.[0])}
        />
      </div>

      {localError && (
        <p className="mt-3 rounded-lg border border-attention-line bg-attention-soft px-4 py-3 text-sm text-body">
          {localError}
        </p>
      )}
    </div>
  );
}
