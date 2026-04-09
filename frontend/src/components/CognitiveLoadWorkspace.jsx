import { useState } from 'react';
import { optimizeCognitiveLoad } from '../api';
import CognitiveLoadResults from './CognitiveLoadResults';

function CognitiveLoadWorkspace() {
  const [explanation, setExplanation] = useState('');
  const [userId, setUserId] = useState('default');
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  const handleSubmit = async () => {
    if (!explanation.trim()) {
      setError('Please enter an explanation to optimize.');
      return;
    }

    setProcessing(true);
    setError('');
    setResults(null);

    try {
      const result = await optimizeCognitiveLoad(explanation, userId);
      setResults(result);
    } catch (err) {
      setError(err.message || 'Cognitive load optimization failed.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Input Form */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-2xl">🧠</span>
          <h2 className="text-lg font-semibold text-gray-900">Cognitive Load Optimizer</h2>
        </div>
        <p className="text-sm text-gray-500 mb-6">
          Submit an explanation to optimize its cognitive load. The system analyzes step density,
          concept gaps, and memory demand, then adapts the explanation structure to match user capacity.
          It does NOT change content — only how reasoning is presented.
        </p>

        {/* Explanation Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Explanation <span className="text-red-500">*</span>
          </label>
          <textarea
            value={explanation}
            onChange={(e) => setExplanation(e.target.value)}
            placeholder="Enter the explanation to optimize for cognitive load..."
            rows={8}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-teal-500 focus:border-teal-500 resize-y"
          />
        </div>

        {/* User ID */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            User ID <span className="text-gray-400 font-normal">(for state tracking)</span>
          </label>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="default"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-teal-500 focus:border-teal-500"
          />
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={processing || !explanation.trim()}
          className={`w-full py-3 px-4 rounded-lg text-sm font-medium transition-colors ${
            processing || !explanation.trim()
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-teal-600 text-white hover:bg-teal-700'
          }`}
        >
          {processing ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Optimizing...
            </span>
          ) : (
            'Optimize Cognitive Load'
          )}
        </button>
      </div>

      {/* Results */}
      {results && <CognitiveLoadResults results={results} />}
    </div>
  );
}

export default CognitiveLoadWorkspace;
