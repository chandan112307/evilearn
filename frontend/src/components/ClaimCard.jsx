import { useState } from 'react';
import { submitFeedback, editClaim } from '../api';
import EvidenceViewer from './EvidenceViewer';

function ClaimCard({ claim, index, sessionId }) {
  const [feedback, setFeedback] = useState(null);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(claim.claim_text);
  const [showEvidence, setShowEvidence] = useState(false);
  const [error, setError] = useState('');

  const statusConfig = {
    supported: {
      color: 'bg-green-50 border-green-200',
      badge: 'bg-green-100 text-green-800',
      icon: '✓',
      label: 'Supported',
    },
    weakly_supported: {
      color: 'bg-yellow-50 border-yellow-200',
      badge: 'bg-yellow-100 text-yellow-800',
      icon: '~',
      label: 'Weakly Supported',
    },
    unsupported: {
      color: 'bg-red-50 border-red-200',
      badge: 'bg-red-100 text-red-800',
      icon: '✗',
      label: 'Unsupported',
    },
  };

  const config = statusConfig[claim.status] || statusConfig.unsupported;

  const handleFeedback = async (decision) => {
    try {
      await submitFeedback(claim.claim_id, sessionId, decision);
      setFeedback(decision);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleEdit = async () => {
    if (!editText.trim()) return;
    try {
      await editClaim(claim.claim_id, sessionId, editText);
      setEditing(false);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  };

  const confidencePercent = Math.round(claim.confidence_score * 100);

  return (
    <div className={`rounded-xl border p-5 ${config.color}`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-500">Claim {index}</span>
          <span className={`px-2.5 py-0.5 text-xs font-medium rounded-full ${config.badge}`}>
            {config.icon} {config.label}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="text-right">
            <div className="text-xs text-gray-500">Confidence</div>
            <div className="text-sm font-semibold">{confidencePercent}%</div>
          </div>
          <div className="w-12 h-12 relative">
            <svg className="w-12 h-12 -rotate-90" viewBox="0 0 36 36">
              <path
                className="text-gray-200"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
              />
              <path
                className={claim.status === 'supported' ? 'text-green-500' : claim.status === 'weakly_supported' ? 'text-yellow-500' : 'text-red-500'}
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeDasharray={`${confidencePercent}, 100`}
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Claim Text */}
      {editing ? (
        <div className="mb-3">
          <textarea
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            rows={2}
            className="w-full border border-gray-300 rounded-lg p-2 text-sm"
          />
          <div className="flex gap-2 mt-2">
            <button onClick={handleEdit} className="px-3 py-1 text-xs bg-blue-600 text-white rounded-md hover:bg-blue-700">
              Re-validate
            </button>
            <button onClick={() => setEditing(false)} className="px-3 py-1 text-xs bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300">
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <p className="text-sm text-gray-800 mb-3">{claim.claim_text}</p>
      )}

      {/* Explanation */}
      {claim.explanation && (
        <div className="bg-white/60 rounded-lg p-3 mb-3">
          <p className="text-xs font-medium text-gray-500 mb-1">Explanation</p>
          <p className="text-sm text-gray-700">{claim.explanation}</p>
        </div>
      )}

      {/* Evidence Toggle */}
      {claim.evidence && claim.evidence.length > 0 && (
        <button
          onClick={() => setShowEvidence(!showEvidence)}
          className="text-xs text-blue-600 hover:text-blue-800 font-medium mb-3"
        >
          {showEvidence ? 'Hide Evidence' : `View Evidence (${claim.evidence.length})`}
        </button>
      )}

      {showEvidence && <EvidenceViewer evidence={claim.evidence} />}

      {/* Actions */}
      <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-200/50">
        {feedback ? (
          <span className={`text-xs font-medium ${feedback === 'accept' ? 'text-green-600' : 'text-red-600'}`}>
            {feedback === 'accept' ? '✓ Accepted' : '✗ Rejected'}
          </span>
        ) : (
          <>
            <button
              onClick={() => handleFeedback('accept')}
              className="px-3 py-1.5 text-xs font-medium bg-green-100 text-green-700 rounded-md hover:bg-green-200 transition-colors"
            >
              Accept
            </button>
            <button
              onClick={() => handleFeedback('reject')}
              className="px-3 py-1.5 text-xs font-medium bg-red-100 text-red-700 rounded-md hover:bg-red-200 transition-colors"
            >
              Reject
            </button>
          </>
        )}
        <button
          onClick={() => setEditing(true)}
          className="px-3 py-1.5 text-xs font-medium bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors ml-auto"
        >
          Edit
        </button>
      </div>

      {error && (
        <div className="mt-2 text-xs text-red-600">{error}</div>
      )}
    </div>
  );
}

export default ClaimCard;
