import { useState } from 'react';
import DocumentUpload from './components/DocumentUpload';
import ValidationWorkspace from './components/ValidationWorkspace';
import ResultsDisplay from './components/ResultsDisplay';
import HistoryDashboard from './components/HistoryDashboard';
import StressTestWorkspace from './components/StressTestWorkspace';
import ThinkingSimulationWorkspace from './components/ThinkingSimulationWorkspace';

function App() {
  const [activeTab, setActiveTab] = useState('workspace');
  const [results, setResults] = useState(null);
  const [sessionId, setSessionId] = useState(null);

  const handleResults = (data) => {
    setResults(data);
    setSessionId(data.session_id);
  };

  const tabs = [
    { id: 'workspace', label: 'Validation Workspace' },
    { id: 'thinking-sim', label: 'Thinking Simulator' },
    { id: 'stress-test', label: 'Stress Test' },
    { id: 'documents', label: 'Documents' },
    { id: 'history', label: 'History' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">EviLearn</h1>
              <p className="text-sm text-gray-500">Claim-based Knowledge Validation</p>
            </div>
            <nav className="flex gap-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? tab.id === 'stress-test'
                        ? 'bg-purple-600 text-white'
                        : tab.id === 'thinking-sim'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-blue-600 text-white'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'documents' && <DocumentUpload />}

        {activeTab === 'workspace' && (
          <div className="space-y-8">
            <ValidationWorkspace onResults={handleResults} />
            {results && (
              <ResultsDisplay results={results} sessionId={sessionId} />
            )}
          </div>
        )}

        {activeTab === 'thinking-sim' && <ThinkingSimulationWorkspace />}

        {activeTab === 'stress-test' && <StressTestWorkspace />}

        {activeTab === 'history' && <HistoryDashboard />}
      </main>
    </div>
  );
}

export default App;
