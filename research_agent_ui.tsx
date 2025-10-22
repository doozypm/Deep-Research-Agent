import React, { useState } from 'react';
import { Search, Users, AlertTriangle, CheckCircle, Database, Network, FileText, Activity } from 'lucide-react';

const DeepResearchAgentUI = () => {
  const [activeTab, setActiveTab] = useState('architecture');

  const architectureData = {
    components: [
      {
        name: 'Query Orchestrator',
        icon: Activity,
        description: 'Manages search progression and strategy adaptation',
        features: ['Dynamic query generation', 'Context tracking', 'Search prioritization']
      },
      {
        name: 'Multi-Model Engine',
        icon: Database,
        description: 'Integrates Claude, Gemini, and GPT models',
        features: ['Model routing', 'Consensus validation', 'Specialized analysis']
      },
      {
        name: 'Fact Extractor',
        icon: Search,
        description: 'Extracts and validates information',
        features: ['Entity recognition', 'Temporal tracking', 'Cross-reference validation']
      },
      {
        name: 'Risk Analyzer',
        icon: AlertTriangle,
        description: 'Identifies patterns and red flags',
        features: ['Anomaly detection', 'Pattern matching', 'Confidence scoring']
      },
      {
        name: 'Connection Mapper',
        icon: Network,
        description: 'Traces relationships and networks',
        features: ['Graph building', 'Relationship scoring', 'Network analysis']
      },
      {
        name: 'Report Generator',
        icon: FileText,
        description: 'Produces comprehensive assessments',
        features: ['Evidence compilation', 'Risk scoring', 'Visualization']
      }
    ]
  };

  const searchStrategy = {
    phases: [
      {
        phase: 1,
        name: 'Initial Discovery',
        queries: ['Basic biographical info', 'Current positions', 'Public presence'],
        models: ['Claude Sonnet 4.5'],
        depth: 'Surface'
      },
      {
        phase: 2,
        name: 'Professional Deep Dive',
        queries: ['Career history', 'Company affiliations', 'Board memberships'],
        models: ['GPT-4.1', 'Gemini 2.5'],
        depth: 'Medium'
      },
      {
        phase: 3,
        name: 'Network Mapping',
        queries: ['Business partners', 'Family connections', 'Associates'],
        models: ['Claude Opus 4', 'GPT-4.1'],
        depth: 'Deep'
      },
      {
        phase: 4,
        name: 'Risk Assessment',
        queries: ['Legal issues', 'Financial irregularities', 'Reputation events'],
        models: ['All models - Consensus'],
        depth: 'Deep'
      },
      {
        phase: 5,
        name: 'Hidden Connections',
        queries: ['Indirect relationships', 'Historical patterns', 'Obscure mentions'],
        models: ['Claude Opus 4', 'Gemini 2.5'],
        depth: 'Maximum'
      }
    ]
  };

  const evaluationSet = [
    {
      name: 'Test Persona 1: Tech Executive',
      hiddenFacts: [
        'Changed name in 2010 (difficulty: hard)',
        'Silent partner in failed startup 2015 (difficulty: very hard)',
        'Family connection to regulatory official (difficulty: extreme)',
        'Unreported board position in offshore entity (difficulty: extreme)',
        'Co-authored paper under pseudonym (difficulty: hard)'
      ]
    },
    {
      name: 'Test Persona 2: Financial Consultant',
      hiddenFacts: [
        'Previous employment at sanctioned firm (difficulty: medium)',
        'Shared address with litigation subject (difficulty: hard)',
        'Indirect investment in competitor (difficulty: very hard)',
        'Undisclosed family business (difficulty: hard)',
        'Name variation in academic records (difficulty: medium)'
      ]
    },
    {
      name: 'Test Persona 3: Nonprofit Director',
      hiddenFacts: [
        'Prior affiliation with controversial org (difficulty: medium)',
        'Funding from politically exposed person (difficulty: hard)',
        'Overlap in board members with questionable entity (difficulty: very hard)',
        'Historical social media under different name (difficulty: hard)',
        'Academic credential discrepancy (difficulty: medium)'
      ]
    }
  ];

  const technicalStack = {
    core: [
      { tech: 'LangGraph', purpose: 'Agent orchestration and state management' },
      { tech: 'Python 3.11+', purpose: 'Primary implementation language' },
      { tech: 'FastAPI', purpose: 'API server and endpoints' },
      { tech: 'Redis', purpose: 'Caching and rate limiting' }
    ],
    aiModels: [
      { model: 'Claude Sonnet 4.5', use: 'Primary reasoning and analysis' },
      { model: 'Claude Opus 4', use: 'Deep analysis and complex reasoning' },
      { model: 'GPT-4.1', use: 'Alternative perspective and validation' },
      { model: 'Gemini 2.5', use: 'Multi-modal analysis and pattern detection' }
    ],
    dataSources: [
      { source: 'Tavily API', type: 'Real-time web search' },
      { source: 'Serper API', type: 'Google search integration' },
      { source: 'Companies House API', type: 'UK corporate data' },
      { source: 'OpenCorporates', type: 'Global company registry' }
    ],
    utilities: [
      { lib: 'NetworkX', purpose: 'Graph analysis and visualization' },
      { lib: 'SpaCy', purpose: 'NLP and entity extraction' },
      { lib: 'Pydantic', purpose: 'Data validation' },
      { lib: 'SQLite/PostgreSQL', purpose: 'Data persistence' }
    ]
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
            Deep Research AI Agent
          </h1>
          <p className="text-slate-300">Autonomous Intelligence Gathering System</p>
        </div>

        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {['architecture', 'strategy', 'evaluation', 'stack'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap ${
                activeTab === tab
                  ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/50'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {activeTab === 'architecture' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {architectureData.components.map((comp, idx) => {
              const Icon = comp.icon;
              return (
                <div key={idx} className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700 hover:border-blue-500 transition-all">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-blue-500/20 rounded-lg">
                      <Icon className="w-6 h-6 text-blue-400" />
                    </div>
                    <h3 className="font-bold text-lg">{comp.name}</h3>
                  </div>
                  <p className="text-slate-300 text-sm mb-4">{comp.description}</p>
                  <div className="space-y-2">
                    {comp.features.map((feature, fidx) => (
                      <div key={fidx} className="flex items-center gap-2 text-sm">
                        <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                        <span className="text-slate-300">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === 'strategy' && (
          <div className="space-y-4">
            {searchStrategy.phases.map((phase, idx) => (
              <div key={idx} className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center font-bold text-xl">
                    {phase.phase}
                  </div>
                  <div>
                    <h3 className="font-bold text-xl">{phase.name}</h3>
                    <p className="text-sm text-slate-400">Depth: {phase.depth}</p>
                  </div>
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-semibold text-blue-400 mb-2">Query Focus</h4>
                    <ul className="space-y-1">
                      {phase.queries.map((q, qidx) => (
                        <li key={qidx} className="text-sm text-slate-300 flex items-start gap-2">
                          <span className="text-blue-400 mt-1">•</span>
                          {q}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-blue-400 mb-2">AI Models</h4>
                    <div className="space-y-1">
                      {phase.models.map((m, midx) => (
                        <div key={midx} className="text-sm bg-slate-700/50 rounded px-3 py-1 text-slate-300">
                          {m}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'evaluation' && (
          <div className="space-y-6">
            {evaluationSet.map((persona, idx) => (
              <div key={idx} className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
                <div className="flex items-center gap-3 mb-4">
                  <Users className="w-6 h-6 text-blue-400" />
                  <h3 className="font-bold text-xl">{persona.name}</h3>
                </div>
                <h4 className="text-sm font-semibold text-slate-400 mb-3">Hidden Facts to Discover:</h4>
                <div className="space-y-2">
                  {persona.hiddenFacts.map((fact, fidx) => {
                    const difficulty = fact.match(/\(difficulty: (.*?)\)/)[1];
                    const factText = fact.replace(/\(difficulty:.*?\)/, '').trim();
                    const difficultyColor = {
                      'medium': 'text-yellow-400',
                      'hard': 'text-orange-400',
                      'very hard': 'text-red-400',
                      'extreme': 'text-purple-400'
                    }[difficulty];
                    return (
                      <div key={fidx} className="flex items-start gap-3 bg-slate-700/30 rounded-lg p-3">
                        <AlertTriangle className={`w-5 h-5 ${difficultyColor} flex-shrink-0 mt-0.5`} />
                        <div className="flex-1">
                          <p className="text-slate-200 text-sm">{factText}</p>
                          <span className={`text-xs ${difficultyColor} font-semibold`}>
                            {difficulty.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'stack' && (
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
              <h3 className="font-bold text-xl mb-4 text-blue-400">Core Technologies</h3>
              <div className="space-y-3">
                {technicalStack.core.map((item, idx) => (
                  <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                    <div className="font-semibold text-slate-200">{item.tech}</div>
                    <div className="text-sm text-slate-400">{item.purpose}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
              <h3 className="font-bold text-xl mb-4 text-blue-400">AI Models</h3>
              <div className="space-y-3">
                {technicalStack.aiModels.map((item, idx) => (
                  <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                    <div className="font-semibold text-slate-200">{item.model}</div>
                    <div className="text-sm text-slate-400">{item.use}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
              <h3 className="font-bold text-xl mb-4 text-blue-400">Data Sources</h3>
              <div className="space-y-3">
                {technicalStack.dataSources.map((item, idx) => (
                  <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                    <div className="font-semibold text-slate-200">{item.source}</div>
                    <div className="text-sm text-slate-400">{item.type}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
              <h3 className="font-bold text-xl mb-4 text-blue-400">Utilities</h3>
              <div className="space-y-3">
                {technicalStack.utilities.map((item, idx) => (
                  <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                    <div className="font-semibold text-slate-200">{item.lib}</div>
                    <div className="text-sm text-slate-400">{item.purpose}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DeepResearchAgentUI;