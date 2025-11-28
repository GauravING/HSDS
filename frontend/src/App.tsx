import { useState } from "react";
import Landing from "./components/Landing";
import UploadCard from "./components/UploadCard";
import ResultModal from "./components/ResultModal";

function App() {
  const [detections, setDetections] = useState<
    { class: string; confidence: number }[]
  >([]);
  const [open, setOpen] = useState(false);

  return (
    <>
      <Landing />
      <section className="pb-24 px-4">
        <UploadCard
          onResult={(d) => {
            setDetections(d);
            setOpen(true);
          }}
        />
        {open && (
          <ResultModal
            detections={detections}
            onClose={() => setOpen(false)}
          />
        )}
      </section>
      <footer className="text-center py-6 text-gray-500 text-sm">
        © 2025 Helmet & Seat-belt AI
      </footer>
    </>
  );
}

export default App;
