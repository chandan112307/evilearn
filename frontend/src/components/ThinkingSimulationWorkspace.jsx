import { useState } from 'react';
import { simulateThinking } from '../api';
import ThinkingSimulationResults from './ThinkingSimulationResults';

function ThinkingSimulationWorkspace() {
  const [problem, setProblem] = useState('');
  const [studentAnswer, setStudentAnswer] = useState('');
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  const handleSubmit = async () => {
    if (!problem.trim()) {
      setError('Please enter a problem or question to analyze.');
      return;
    }

    setProcessing(true);
    setError('');
    setResults(null);

    try {
      const result = await simulateThinking(problem, studentAnswer);
      setResults(result);
    } catch (err) {
      setError(err.message || 'Thinking simulation failed.');
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
          <h2 className="text-lg font-semibold text-gray-900">Thinking Simulation Engine</h2>
        </div>
        <p className="text-sm text-gray-500 mb-6">
          Simulate how different cognitive levels (beginner, intermediate, expert) reason about a problem.
          The system analyzes reasoning structure, strategy differences, and abstraction levels —
          it does NOT solve problems or check correctness.
          Optionally provide a student answer to identify thinking gaps.
        </p>

        {/* Problem (mandatory) */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Problem / Question <span className="text-red-500">*</span>
          </label>
          <textarea
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            placeholder="Enter the problem, question, or concept to simulate reasoning for..."
            rows={4}
            className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-y"
            disabled={processing}
          />
        </div>

        {/* Student Answer (optional) */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Student Answer / Reasoning <span className="text-gray-400 font-normal">(optional)</span>
          </label>
          <textarea
            value={studentAnswer}
            onChange={(e) => setStudentAnswer(e.target.value)}
            placeholder="Enter the student's reasoning to compare against simulated thinking levels..."
            rows={4}
            className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-y"
            disabled={processing}
          />
          <p className="text-xs text-gray-400 mt-1">
            If provided, the system will compare the student's reasoning against all three cognitive levels and identify gaps.
          </p>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={handleSubmit}
            disabled={processing || !problem.trim()}
            className={`px-6 py-2.5 rounded-lg text-sm font-medium text-white transition-colors ${
              processing || !problem.trim()
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            {processing ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Simulating Thinking...
              </span>
            ) : (
              '🧠 Simulate Thinking'
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {results && <ThinkingSimulationResults results={results} />}
    </div>
  );
}

export default ThinkingSimulationWorkspace;
