function EvidenceViewer({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic mb-3">No evidence available.</div>
    );
  }

  return (
    <div className="space-y-2 mb-3">
      {evidence.map((item, i) => (
        <div key={i} className="bg-white rounded-lg border border-gray-200 p-3">
          <div className="flex items-center gap-2 mb-1.5">
            <svg className="h-3.5 w-3.5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="text-xs font-medium text-blue-600">
              Page {item.page_number}
            </span>
          </div>
          <p className="text-xs text-gray-700 leading-relaxed">{item.snippet}</p>
        </div>
      ))}
    </div>
  );
}

export default EvidenceViewer;
