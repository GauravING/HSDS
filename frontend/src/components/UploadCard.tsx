// @ts-nocheck
import { useRef, useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import { buildUrl } from "../utils/api";

// Use buildUrl to resolve base (env override or relative) and target the upload endpoint
const UPLOAD_URL = buildUrl("/detect/upload");

interface Detection {
  class: string;
  confidence: number;
  box: number[];
}

export default function UploadCard({
  onResult,
}: {
  onResult: (d: Detection[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  // Hooks must be at top-level of component
  const auth: any = useAuth();

  const sendFile = async (file: File) => {
    setBusy(true);
  const form = new FormData();
  form.append("file", file);
  // Use authFetch if available so the JWT is attached to the request
  const fetcher = auth && auth.authFetch ? auth.authFetch : fetch;
  const res = await fetcher(UPLOAD_URL, { method: "POST", body: form });
  const detections = (await res.json()) as Detection[];
    onResult(detections);
    setBusy(false);
  };

  return (
    <div
      className="mx-auto max-w-md bg-gray-800 rounded-xl p-6 shadow-lg"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        if (e.dataTransfer.files[0]) sendFile(e.dataTransfer.files[0]);
      }}
    >
      <input
        type="file"
        accept="image/*"
        ref={inputRef}
        hidden
        onChange={(e) => e.target.files && sendFile(e.target.files[0])}
      />
      <button
        onClick={() => inputRef.current?.click()}
        className="w-full py-3 text-lg font-semibold rounded-lg bg-emerald-500 hover:bg-emerald-600 transition"
      >
        {busy ? "Detecting…" : "Select or Drop Image"}
      </button>
      <p className="mt-2 text-sm text-gray-400 text-center">
        JPG • PNG • WEBP up to 5 MB
      </p>
    </div>
  );
}
