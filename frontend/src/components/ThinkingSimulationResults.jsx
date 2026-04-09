function ThinkingSimulationResults({ results }) {
  if (!results) return null;

  const {
    cognitive_profiles = [],
    reasoning_paths = [],
    strategy_tags = [],
    comparison_results = {},
    gap_analysis = [],
    student_comparison = {},
  } = results;

  const levelColors = {
    beginner: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-800',
      badge: 'bg-blue-100 text-blue-800',
      accent: 'bg-blue-500',
    },
    intermediate: {
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      text: 'text-amber-800',
      badge: 'bg-amber-100 text-amber-800',
      accent: 'bg-amber-500',
    },
    expert: {
      bg: 'bg-emerald-50',
      border: 'border-emerald-200',
      text: 'text-emerald-800',
      badge: 'bg-emerald-100 text-emerald-800',
      accent: 'bg-emerald-500',
    },
  };

  const levelIcons = {
    beginner: '🌱',
    intermediate: '🔧',
    expert: '🎯',
  };

  const severityConfig = {
    info: { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-800', icon: 'ℹ️' },
    warning: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-800', icon: '⚠️' },
    critical: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-800', icon: '🔴' },
  };

  const hasStudentComparison = student_comparison && Object.keys(student_comparison).length > 0
    && student_comparison.student_level_match && student_comparison.student_level_match !== 'unknown';

  return (
    <div className="space-y-6">
      {/* Cognitive Profiles */}
      {cognitive_profiles.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>🧠</span>
            Cognitive Profiles
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {cognitive_profiles.map((profile, i) => {
              const colors = levelColors[profile.level] || levelColors.beginner;
              const icon = levelIcons[profile.level] || '📝';
              return (
                <div key={i} className={`rounded-lg border p-4 ${colors.bg} ${colors.border}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">{icon}</span>
                    <span className={`px-2 py-0.5 text-xs font-semibold rounded-full ${colors.badge}`}>
                      {profile.level.charAt(0).toUpperCase() + profile.level.slice(1)}
                    </span>
                  </div>
                  <p className={`text-sm mb-3 ${colors.text}`}>{profile.description}</p>
                  {profile.characteristics && profile.characteristics.length > 0 && (
                    <ul className="space-y-1">
                      {profile.characteristics.map((char, j) => (
                        <li key={j} className={`text-xs ${colors.text} flex items-start gap-1.5`}>
                          <span className="mt-0.5">•</span>
                          <span>{char}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Reasoning Paths */}
      {reasoning_paths.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>🔀</span>
            Reasoning Paths
          </h3>
          <div className="space-y-6">
            {reasoning_paths.map((path, i) => {
              const colors = levelColors[path.level] || levelColors.beginner;
              const icon = levelIcons[path.level] || '📝';
              return (
                <div key={i} className={`rounded-lg border p-4 ${colors.bg} ${colors.border}`}>
                  <div className="flex items-center gap-2 mb-4">
                    <span className="text-lg">{icon}</span>
                    <span className={`px-2 py-0.5 text-xs font-semibold rounded-full ${colors.badge}`}>
                      {path.level.charAt(0).toUpperCase() + path.level.slice(1)} Path
                    </span>
                    <span className="text-xs text-gray-500">
                      {path.steps.length} step{path.steps.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                  <div className="space-y-3">
                    {path.steps.map((step, j) => (
                      <div key={j} className="bg-white rounded-lg p-3 border border-gray-100 shadow-sm">
                        <div className="flex items-center gap-2 mb-2">
                          <div className={`w-6 h-6 rounded-full ${colors.accent} text-white text-xs flex items-center justify-center font-bold`}>
                            {j + 1}
                          </div>
                          <span className="text-sm font-medium text-gray-900">
                            {step.operation_type.replace(/_/g, ' ')}
                          </span>
                          {step.concept_used && (
                            <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full">
                              {step.concept_used}
                            </span>
                          )}
                        </div>
                        {(step.input_value || step.output_value) && (
                          <div className="text-xs text-gray-600 ml-8 mb-1">
                            {step.input_value && <span className="block"><span className="font-medium">Input:</span> {step.input_value}</span>}
                            {step.output_value && <span className="block"><span className="font-medium">Output:</span> {step.output_value}</span>}
                          </div>
                        )}
                        {step.reason && (
                          <p className="text-xs text-gray-500 ml-8 italic">{step.reason}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Strategy Tags */}
      {strategy_tags.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>🏷️</span>
            Strategy Tags
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {strategy_tags.map((tagEntry, i) => {
              const colors = levelColors[tagEntry.level] || levelColors.beginner;
              return (
                <div key={i} className={`rounded-lg border p-4 ${colors.bg} ${colors.border}`}>
                  <span className={`text-sm font-semibold ${colors.text}`}>
                    {tagEntry.level.charAt(0).toUpperCase() + tagEntry.level.slice(1)}
                  </span>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {tagEntry.tags.map((tag, j) => (
                      <span key={j} className={`px-2 py-1 text-xs rounded-full ${colors.badge}`}>
                        {tag.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Comparative Analysis */}
      {comparison_results && Object.keys(comparison_results).length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>📊</span>
            Comparative Analysis
          </h3>

          {/* Structural Comparison */}
          {comparison_results.structural && Object.keys(comparison_results.structural).length > 0 && (
            <div className="mb-6">
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Structural Comparison</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {Object.entries(comparison_results.structural).map(([level, data]) => {
                  const colors = levelColors[level] || levelColors.beginner;
                  return (
                    <div key={level} className={`rounded-lg border p-3 ${colors.bg} ${colors.border}`}>
                      <span className={`text-sm font-medium ${colors.text}`}>
                        {level.charAt(0).toUpperCase() + level.slice(1)}
                      </span>
                      <div className="mt-2 space-y-1">
                        <div className="text-xs text-gray-600">
                          <span className="font-medium">Steps:</span> {data.step_count}
                        </div>
                        <div className="text-xs text-gray-600">
                          <span className="font-medium">Approach:</span>{' '}
                          {data.is_linear ? 'Linear' : 'Transformed'}
                        </div>
                        <div className="text-xs text-gray-600">
                          <span className="font-medium">Abstraction:</span>{' '}
                          {data.abstraction_level}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Key Differences */}
          {comparison_results.key_differences && comparison_results.key_differences.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Key Differences</h4>
              <div className="space-y-2">
                {comparison_results.key_differences.map((diff, i) => (
                  <div key={i} className="p-3 bg-gray-50 rounded-lg text-sm text-gray-700 flex items-start gap-2">
                    <span className="font-bold text-indigo-500 mt-0.5">→</span>
                    <span>{diff}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Student Comparison */}
      {hasStudentComparison && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>👤</span>
            Student Comparison
          </h3>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-600">Your reasoning matches:</span>
              <span className={`px-3 py-1 text-sm font-semibold rounded-full ${
                (levelColors[student_comparison.student_level_match] || levelColors.beginner).badge
              }`}>
                {student_comparison.student_level_match.charAt(0).toUpperCase() +
                 student_comparison.student_level_match.slice(1)} Level
              </span>
            </div>

            {student_comparison.missing_steps && student_comparison.missing_steps.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Missing Steps</h4>
                <div className="space-y-1">
                  {student_comparison.missing_steps.map((step, i) => (
                    <div key={i} className="p-2 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-800">
                      ⚠️ {step}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {student_comparison.missing_strategies && student_comparison.missing_strategies.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Missing Strategies</h4>
                <div className="flex flex-wrap gap-2">
                  {student_comparison.missing_strategies.map((strategy, i) => (
                    <span key={i} className="px-2 py-1 text-xs bg-red-50 text-red-700 border border-red-200 rounded-full">
                      {strategy.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {student_comparison.inefficiencies && student_comparison.inefficiencies.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Inefficiencies</h4>
                <div className="space-y-1">
                  {student_comparison.inefficiencies.map((ineff, i) => (
                    <div key={i} className="p-2 bg-orange-50 border border-orange-200 rounded text-sm text-orange-800">
                      {ineff}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {student_comparison.abstraction_gaps && student_comparison.abstraction_gaps.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Abstraction Gaps</h4>
                <div className="space-y-1">
                  {student_comparison.abstraction_gaps.map((gap, i) => (
                    <div key={i} className="p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
                      🔴 {gap}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Gap Analysis */}
      {gap_analysis.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>🔍</span>
            Thinking Gap Analysis
          </h3>
          <div className="space-y-2">
            {gap_analysis.map((gap, i) => {
              const config = severityConfig[gap.severity] || severityConfig.info;
              return (
                <div
                  key={i}
                  className={`p-3 rounded-lg border text-sm ${config.bg} ${config.border} ${config.text}`}
                >
                  <span className="mr-2">{config.icon}</span>
                  {gap.insight}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default ThinkingSimulationResults;
