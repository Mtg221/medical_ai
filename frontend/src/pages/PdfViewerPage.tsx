import { useEffect, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export function PdfViewerPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [pdfDoc, setPdfDoc] = useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [pageNum, setPageNum] = useState<number>(1);
  const [numPages, setNumPages] = useState<number>(0);
  const [scale, setScale] = useState<number>(1.25);
  const [extractedText, setExtractedText] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");

  const handleFileChange = async (file: File | undefined) => {
    if (!file) return;
    setErrorMsg("");
    try {
      const buf = await file.arrayBuffer();
      const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(buf) });
      const doc = await loadingTask.promise;
      setPdfDoc(doc);
      setNumPages(doc.numPages);
      setPageNum(1);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to load PDF");
    }
  };

  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;
    let cancelled = false;

    const renderPage = async () => {
      const page = await pdfDoc.getPage(pageNum);
      if (cancelled || !canvasRef.current) return;
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      await page.render({ canvasContext: ctx, viewport, canvas }).promise;
      const content = await page.getTextContent();
      const text = content.items
        .map((item) => ("str" in item ? item.str : ""))
        .join(" ");
      if (!cancelled) setExtractedText(text);
    };

    void renderPage();
    return () => {
      cancelled = true;
    };
  }, [pdfDoc, pageNum, scale]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-semibold text-slate-900">PDF.js Document & OCR Inspector</h1>
            <p className="text-xs text-slate-500">Local client-side PDF rendering via pdfjs-dist with page text layer inspection</p>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="file"
              accept="application/pdf"
              aria-label="Open local PDF"
              onChange={(e) => void handleFileChange(e.target.files?.[0])}
              className="text-xs text-slate-600"
            />
            {numPages > 0 && (
              <>
                <Button size="sm" variant="outline" onClick={() => setPageNum((p) => Math.max(1, p - 1))} disabled={pageNum <= 1}>
                  <ChevronLeft className="h-3.5 w-3.5" />
                </Button>
                <span className="text-xs text-slate-700">
                  Page {pageNum} / {numPages}
                </span>
                <Button size="sm" variant="outline" onClick={() => setPageNum((p) => Math.min(numPages, p + 1))} disabled={pageNum >= numPages}>
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
                <Button size="sm" variant="outline" onClick={() => setScale((s) => Math.max(0.5, s - 0.25))}>
                  <ZoomOut className="h-3.5 w-3.5" />
                </Button>
                <Button size="sm" variant="outline" onClick={() => setScale((s) => Math.min(2.5, s + 0.25))}>
                  <ZoomIn className="h-3.5 w-3.5" />
                </Button>
              </>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {errorMsg && <p className="text-xs text-rose-600 mb-3">{errorMsg}</p>}
          {!pdfDoc ? (
            <div className="rounded border border-dashed border-slate-300 bg-slate-50 p-10 text-center text-xs text-slate-500">
              Select a local digital or scanned PDF file above to render pages with pdf.js.
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="overflow-auto border border-slate-200 rounded bg-slate-100 p-3 flex justify-center">
                <canvas ref={canvasRef} />
              </div>
              <div className="border border-slate-200 rounded bg-white p-3 text-xs space-y-2">
                <div className="font-semibold text-slate-900">Embedded Text Layer (Page {pageNum})</div>
                <pre className="whitespace-pre-wrap text-slate-700 font-mono text-[11px]">
                  {extractedText || "(No embedded text layer — image-only scan; upload via Upload & Jobs for Tesseract OCR.)"}
                </pre>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
