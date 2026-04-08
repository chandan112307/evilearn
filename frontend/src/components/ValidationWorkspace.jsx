import { useState } from 'react';
import { processInput } from '../api';

function ValidationWorkspace({ onResults }) {
  const [inputText, setInputText] = useState('');
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!inputText.trim()) {
      setError('Please enter text to validate.');
      return;
    }

    setProcessing(true);
    setError('');

    try {
      const result = await processInput(inputText);
      onResults(result);
    } catch (err) {
      setError(err.message || 'Validation failed.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-2">Validation Workspace</h2>
      <p className="text-sm text-gray-500 mb-4">
        Enter an answer, summary, or explanation to validate against your uploaded documents.
      </p>

      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Enter your answer, summary, or concept explanation here..."
        rows={6}
        className="w-full border border-gray-300 rounded-lg p-4 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y"
        disabled={processing}
      />

      {error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="mt-4 flex justify-end">
        <button
          onClick={handleSubmit}
          disabled={processing || !inputText.trim()}
          className={`px-6 py-2.5 rounded-lg text-sm font-medium text-white transition-colors ${
            processing || !inputText.trim()
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {processing ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Validating...
            </span>
          ) : (
            'Validate'
          )}
        </button>
      </div>
    </div>
  );
}

export default ValidationWorkspace;
