import { useState, useEffect } from 'react';
import { getHistory } from '../api';

function HistoryDashboard() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedSession, setSelectedSession] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await getHistory();
      setSessions(data.sessions || []);
    } catch {
      // Handle silently
    } finally {
      setLoading(false);
    }
  };

  const statusBadge = (status) => {
    const colors = {
      supported: 'bg-green-100 text-green-800',
      weakly_supported: 'bg-yellow-100 text-yellow-800',
      unsupported: 'bg-red-100 text-red-800',
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
        <svg className="animate-spin h-8 w-8 mx-auto text-blue-600" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <p className="mt-2 text-sm text-gray-500">Loading history...</p>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
        <p className="text-gray-500">No validation history yet.</p>
        <p className="text-sm text-gray-400 mt-1">Submit text for validation to start building your history.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Validation History</h2>

        <div className="space-y-3">
          {sessions.map((session) => {
            const isSelected = selectedSession === session.session_id;
            const results = session.results || [];
            const supported = results.filter((r) => r.status === 'supported').length;
            const weak = results.filter((r) => r.status === 'weakly_supported').length;
            const unsupported = results.filter((r) => r.status === 'unsupported').length;

            return (
              <div key={session.session_id} className="border border-gray-200 rounded-lg overflow-hidden">
                <button
                  onClick={() => setSelectedSession(isSelected ? null : session.session_id)}
                  className="w-full p-4 text-left hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {session.input_text}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {new Date(session.created_at).toLocaleString()} · {results.length} claims
                      </p>
                    </div>
                    <div className="flex gap-2 ml-4">
                      {supported > 0 && (
                        <span className="px-2 py-0.5 text-xs bg-green-100 text-green-800 rounded-full">
                          {supported} ✓
                        </span>
                      )}
                      {weak > 0 && (
                        <span className="px-2 py-0.5 text-xs bg-yellow-100 text-yellow-800 rounded-full">
                          {weak} ~
                        </span>
                      )}
                      {unsupported > 0 && (
                        <span className="px-2 py-0.5 text-xs bg-red-100 text-red-800 rounded-full">
                          {unsupported} ✗
                        </span>
                      )}
                    </div>
                  </div>
                </button>

                {isSelected && results.length > 0 && (
                  <div className="border-t border-gray-200 p-4 bg-gray-50 space-y-3">
                    {results.map((result, i) => (
                      <div key={i} className="bg-white rounded-lg p-3 border border-gray-100">
                        <div className="flex items-start justify-between mb-1">
                          <p className="text-sm text-gray-800 flex-1">{result.claim_text}</p>
                          <span className={`px-2 py-0.5 text-xs font-medium rounded-full ml-2 whitespace-nowrap ${statusBadge(result.status)}`}>
                            {result.status?.replace('_', ' ')}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500">
                          Confidence: {Math.round((result.confidence_score || 0) * 100)}%
                        </p>
                        {result.explanation && (
                          <p className="text-xs text-gray-600 mt-1">{result.explanation}</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default HistoryDashboard;
