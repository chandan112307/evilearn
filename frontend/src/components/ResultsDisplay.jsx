import ClaimCard from './ClaimCard';

function ResultsDisplay({ results, sessionId }) {
  if (!results || !results.claims || results.claims.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center">
        <p className="text-gray-500">
          {results?.message || 'No claims were extracted from the input.'}
        </p>
      </div>
    );
  }

  const statusCounts = results.claims.reduce(
    (acc, c) => {
      acc[c.status] = (acc[c.status] || 0) + 1;
      return acc;
    },
    {}
  );

  return (
    <div className="space-y-6">
      {/* Summary Bar */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">
            Validation Results
          </h3>
          <div className="flex gap-4 text-sm">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
              Supported: {statusCounts.supported || 0}
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
              Weak: {statusCounts.weakly_supported || 0}
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
              Unsupported: {statusCounts.unsupported || 0}
            </span>
          </div>
        </div>
      </div>

      {/* Claim Cards */}
      <div className="space-y-4">
        {results.claims.map((claim, index) => (
          <ClaimCard
            key={claim.claim_id}
            claim={claim}
            index={index + 1}
            sessionId={sessionId}
          />
        ))}
      </div>
    </div>
  );
}

export default ResultsDisplay;
