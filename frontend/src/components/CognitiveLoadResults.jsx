function CognitiveLoadResults({ results }) {
  if (!results) return null;

  const {
    adapted_explanation = [],
    load_state = 'optimal',
    control_actions = [],
    user_state = {},
    load_metrics = {},
    reasoning_mode = 'medium',
  } = results;

  const loadStateColors = {
    overload: 'bg-red-100 text-red-800 border-red-300',
    optimal: 'bg-green-100 text-green-800 border-green-300',
    underload: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  };

  const modeColors = {
    'fine-grained': 'bg-blue-100 text-blue-800',
    medium: 'bg-gray-100 text-gray-800',
    coarse: 'bg-purple-100 text-purple-800',
  };

  const loadStateIcons = {
    overload: '🔴',
    optimal: '🟢',
    underload: '🟡',
  };

  return (
    <div className="space-y-6">
      {/* Status Badges */}
      <div className="flex flex-wrap gap-3">
        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium border ${loadStateColors[load_state] || loadStateColors.optimal}`}>
          {loadStateIcons[load_state] || '🟢'} Load: {load_state}
        </span>
        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${modeColors[reasoning_mode] || modeColors.medium}`}>
          🎛️ Mode: {reasoning_mode}
        </span>
      </div>

      {/* Cognitive Load Metrics */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-md font-semibold text-gray-900 mb-4">📊 Cognitive Load Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">{load_metrics.step_density?.toFixed(2) ?? '0.00'}</p>
            <p className="text-xs text-gray-500 mt-1">Step Density</p>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">{load_metrics.concept_gap?.toFixed(2) ?? '0.00'}</p>
            <p className="text-xs text-gray-500 mt-1">Concept Gap</p>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">{load_metrics.memory_demand?.toFixed(2) ?? '0.00'}</p>
            <p className="text-xs text-gray-500 mt-1">Memory Demand</p>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">{load_metrics.total_load?.toFixed(2) ?? '0.00'}</p>
            <p className="text-xs text-gray-500 mt-1">Total Load</p>
          </div>
        </div>
      </div>

      {/* Control Actions */}
      {control_actions.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-md font-semibold text-gray-900 mb-3">⚙️ Adaptation Actions</h3>
          <div className="space-y-2">
            {control_actions.map((action, idx) => (
              <div
                key={idx}
                className="flex items-start gap-2 p-3 bg-blue-50 rounded-lg border border-blue-100"
              >
                <span className="text-blue-500 mt-0.5">→</span>
                <div>
                  <span className="text-sm font-medium text-blue-900">{action.action}</span>
                  <p className="text-sm text-blue-700 mt-0.5">{action.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Adapted Explanation */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-md font-semibold text-gray-900 mb-4">📝 Adapted Explanation</h3>
        {adapted_explanation.length === 0 ? (
          <p className="text-sm text-gray-500">No steps generated.</p>
        ) : (
          <div className="space-y-3">
            {adapted_explanation.map((step, idx) => {
              const isCheckpoint = step.step_id?.startsWith('checkpoint');
              const absColors = {
                concrete: 'border-l-green-400',
                'semi-abstract': 'border-l-yellow-400',
                abstract: 'border-l-red-400',
              };
              return (
                <div
                  key={idx}
                  className={`border-l-4 ${isCheckpoint ? 'border-l-blue-400 bg-blue-50' : absColors[step.abstraction_level] || 'border-l-gray-300'} pl-4 py-2`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-mono text-gray-400">{step.step_id}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      step.abstraction_level === 'abstract'
                        ? 'bg-red-100 text-red-700'
                        : step.abstraction_level === 'semi-abstract'
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {step.abstraction_level}
                    </span>
                  </div>
                  <p className="text-sm text-gray-800">{step.content}</p>
                  {step.concepts?.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {step.concepts.map((c, i) => (
                        <span key={i} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                          {c}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* User Cognitive State */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-md font-semibold text-gray-900 mb-4">👤 User Cognitive State</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <p className="text-xs text-gray-500 mb-1">Understanding</p>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full"
                style={{ width: `${(user_state.understanding_level ?? 0.5) * 100}%` }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-1">{((user_state.understanding_level ?? 0.5) * 100).toFixed(0)}%</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Stability</p>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${(user_state.reasoning_stability ?? 0.5) * 100}%` }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-1">{((user_state.reasoning_stability ?? 0.5) * 100).toFixed(0)}%</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Learning Speed</p>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-purple-500 h-2 rounded-full"
                style={{ width: `${(user_state.learning_speed ?? 0.5) * 100}%` }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-1">{((user_state.learning_speed ?? 0.5) * 100).toFixed(0)}%</p>
          </div>
        </div>
        <div className="flex gap-4 mt-4 text-xs text-gray-500">
          <span>Interactions: {user_state.interaction_count ?? 0}</span>
          <span>Overload Signals: {user_state.overload_signals ?? 0}</span>
        </div>
      </div>
    </div>
  );
}

export default CognitiveLoadResults;
