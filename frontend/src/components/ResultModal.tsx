interface Props {
  detections: { class: string; confidence: number }[];
  onClose: () => void;
}
export default function ResultModal({ detections, onClose }: Props) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 p-8 rounded-lg w-full max-w-lg">
        <h2 className="text-2xl font-bold mb-4">Detection Results</h2>
        <ul className="space-y-2 max-h-72 overflow-y-auto pr-2">
          {detections.map((d, i) => (
            <li
              key={i}
              className="flex justify-between items-center bg-gray-800 rounded px-4 py-2"
            >
              <span className="capitalize">{d.class.replace(/_/g, " ")}</span>
              <span className="text-sm text-gray-400">
                {(d.confidence * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
        <button
          onClick={onClose}
          className="mt-6 w-full py-2 bg-blue-600 hover:bg-blue-700 rounded"
        >
          Close
        </button>
      </div>
    </div>
  );
}
