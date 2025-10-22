"""
Deep Research AI Agent - Complete Implementation
A sophisticated autonomous research system for comprehensive entity investigation
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from enum import Enum
import anthropic
import openai
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import httpx
import networkx as nx
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class SearchDepth(Enum):
    SURFACE = "surface"
    MEDIUM = "medium"
    DEEP = "deep"
    MAXIMUM = "maximum"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Fact:
    """Represents a discovered fact with metadata"""
    content: str
    source: str
    confidence: float
    timestamp: datetime
    category: str
    verified: bool = False
    cross_references: List[str] = field(default_factory=list)


@dataclass
class Entity:
    """Represents a person or organization"""
    name: str
    entity_type: str
    facts: List[Fact] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class SearchQuery:
    """Represents a search query with metadata"""
    query: str
    phase: int
    depth: SearchDepth
    parent_query: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class AgentState(TypedDict):
    """State passed between nodes in the graph"""
    target_name: str
    current_phase: int
    executed_queries: List[SearchQuery]
    discovered_facts: List[Fact]
    entities: Dict[str, Entity]
    connections_graph: Dict[str, List[str]]
    risk_assessment: Dict[str, Any]
    next_queries: List[SearchQuery]
    final_report: Optional[str]
    error_log: List[str]


# ============================================================================
# MODEL MANAGERS
# ============================================================================

class ModelManager:
    """Manages multiple AI models for different tasks"""
    
    def __init__(self):
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Note: Gemini requires google-generativeai library
        self.rate_limiters = {
            "claude": {"calls": 0, "reset_time": datetime.now()},
            "openai": {"calls": 0, "reset_time": datetime.now()},
            "gemini": {"calls": 0, "reset_time": datetime.now()}
        }
    
    async def query_claude(self, prompt: str, model: str = "claude-sonnet-4-5-20250929") -> str:
        """Query Claude models"""
        try:
            await self._check_rate_limit("claude")
            response = await asyncio.to_thread(
                self.claude_client.messages.create,
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude query error: {e}")
            return f"Error: {str(e)}"
    
    async def query_openai(self, prompt: str, model: str = "gpt-4-turbo-preview") -> str:
        """Query OpenAI models"""
        try:
            await self._check_rate_limit("openai")
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI query error: {e}")
            return f"Error: {str(e)}"
    
    async def consensus_query(self, prompt: str) -> Dict[str, str]:
        """Query multiple models and return consensus"""
        results = await asyncio.gather(
            self.query_claude(prompt),
            self.query_openai(prompt),
            return_exceptions=True
        )
        
        return {
            "claude": results[0] if not isinstance(results[0], Exception) else str(results[0]),
            "openai": results[1] if not isinstance(results[1], Exception) else str(results[1])
        }
    
    async def _check_rate_limit(self, model: str, limit: int = 50):
        """Simple rate limiting"""
        limiter = self.rate_limiters[model]
        now = datetime.now()
        
        if (now - limiter["reset_time"]).seconds > 60:
            limiter["calls"] = 0
            limiter["reset_time"] = now
        
        if limiter["calls"] >= limit:
            wait_time = 60 - (now - limiter["reset_time"]).seconds
            logger.info(f"Rate limit reached for {model}, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            limiter["calls"] = 0
            limiter["reset_time"] = datetime.now()
        
        limiter["calls"] += 1


# ============================================================================
# SEARCH ENGINE
# ============================================================================

class SearchEngine:
    """Handles web searches using multiple APIs"""
    
    def __init__(self):
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.serper_key = os.getenv("SERPER_API_KEY")
        self.search_cache = {}
    
    async def search_tavily(self, query: str, depth: str = "basic") -> List[Dict]:
        """Search using Tavily API"""
        if query in self.search_cache:
            return self.search_cache[query]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_key,
                        "query": query,
                        "search_depth": depth,
                        "max_results": 10
                    },
                    timeout=30.0
                )
                results = response.json().get("results", [])
                self.search_cache[query] = results
                return results
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []
    
    async def search_serper(self, query: str) -> List[Dict]:
        """Search using Serper API"""
        if query in self.search_cache:
            return self.search_cache[query]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query},
                    headers={"X-API-KEY": self.serper_key},
                    timeout=30.0
                )
                results = response.json().get("organic", [])
                self.search_cache[query] = results
                return results
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []
    
    async def multi_source_search(self, query: str, depth: SearchDepth) -> List[Dict]:
        """Search across multiple sources"""
        depth_map = {
            SearchDepth.SURFACE: "basic",
            SearchDepth.MEDIUM: "basic",
            SearchDepth.DEEP: "advanced",
            SearchDepth.MAXIMUM: "advanced"
        }
        
        results = await asyncio.gather(
            self.search_tavily(query, depth_map[depth]),
            self.search_serper(query),
            return_exceptions=True
        )
        
        combined = []
        for result_set in results:
            if isinstance(result_set, list):
                combined.extend(result_set)
        
        return combined[:15]  # Limit to top 15 results


# ============================================================================
# FACT EXTRACTOR
# ============================================================================

class FactExtractor:
    """Extracts and validates facts from search results"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    async def extract_facts(self, query: str, search_results: List[Dict], 
                           context: Dict[str, Any]) -> List[Fact]:
        """Extract structured facts from search results"""
        
        # Prepare context for the model
        results_text = "\n\n".join([
            f"Source: {r.get('url', 'Unknown')}\nTitle: {r.get('title', 'N/A')}\n"
            f"Content: {r.get('content', r.get('snippet', ''))}"
            for r in search_results[:10]
        ])
        
        extraction_prompt = f"""You are an expert fact extractor for intelligence gathering.

Query: {query}
Context: {json.dumps(context, indent=2)}

Search Results:
{results_text}

Extract all relevant facts following these rules:
1. Focus on biographical details, professional history, affiliations, and connections
2. Note any discrepancies or concerning patterns
3. Identify temporal information (dates, timelines)
4. Flag potential red flags or risks
5. Cross-reference information when possible

Return a JSON array of facts with this structure:
[
  {{
    "content": "The specific fact",
    "source": "URL or source identifier",
    "confidence": 0.0-1.0,
    "category": "biographical|professional|financial|legal|connection|risk",
    "temporal_info": "date/time if applicable",
    "red_flag": boolean
  }}
]

Be precise and factual. Only include verifiable information."""

        response = await self.model_manager.query_claude(extraction_prompt)
        
        try:
            # Parse JSON response
            facts_data = json.loads(response)
            facts = []
            
            for fact_dict in facts_data:
                fact = Fact(
                    content=fact_dict["content"],
                    source=fact_dict["source"],
                    confidence=fact_dict["confidence"],
                    timestamp=datetime.now(),
                    category=fact_dict["category"]
                )
                facts.append(fact)
            
            return facts
        except json.JSONDecodeError:
            logger.error("Failed to parse facts from model response")
            return []
    
    async def verify_fact(self, fact: Fact, all_facts: List[Fact]) -> bool:
        """Cross-reference and verify a fact"""
        similar_facts = [
            f for f in all_facts 
            if f.category == fact.category and f != fact
        ]
        
        if len(similar_facts) >= 2:
            verification_prompt = f"""Cross-reference this fact with similar findings:

Main Fact: {fact.content}
Source: {fact.source}
Confidence: {fact.confidence}

Similar Facts:
{chr(10).join([f"- {f.content} (from {f.source})" for f in similar_facts[:5]])}

Is the main fact corroborated by the similar facts? Respond with JSON:
{{"verified": true/false, "reasoning": "explanation"}}"""

            response = await self.model_manager.query_claude(verification_prompt)
            try:
                result = json.loads(response)
                return result.get("verified", False)
            except:
                return False
        
        return fact.confidence > 0.7


# ============================================================================
# RISK ANALYZER
# ============================================================================

class RiskAnalyzer:
    """Identifies risk patterns and red flags"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.risk_patterns = {
            "legal": ["lawsuit", "indictment", "conviction", "sanction", "violation"],
            "financial": ["bankruptcy", "fraud", "embezzlement", "undisclosed", "conflict"],
            "reputational": ["scandal", "controversy", "resignation", "misconduct"],
            "compliance": ["regulatory", "violation", "penalty", "investigation"],
            "network": ["sanctioned entity", "criminal", "politically exposed"]
        }
    
    async def analyze_risks(self, entity: Entity, all_facts: List[Fact]) -> Dict[str, Any]:
        """Comprehensive risk analysis"""
        
        # Pattern-based risk detection
        detected_patterns = defaultdict(list)
        for fact in entity.facts:
            for category, keywords in self.risk_patterns.items():
                if any(kw.lower() in fact.content.lower() for kw in keywords):
                    detected_patterns[category].append(fact.content)
        
        # AI-powered risk assessment
        facts_summary = "\n".join([f"- {f.content} (confidence: {f.confidence})" 
                                   for f in entity.facts])
        
        risk_prompt = f"""Conduct a comprehensive risk assessment for this entity:

Entity: {entity.name}
Type: {entity.entity_type}

Known Facts:
{facts_summary}

Detected Pattern Categories:
{json.dumps(dict(detected_patterns), indent=2)}

Provide a detailed risk assessment with:
1. Overall risk level (LOW, MEDIUM, HIGH, CRITICAL)
2. Specific risk factors
3. Risk score (0-100)
4. Recommended actions
5. Confidence level

Return JSON format:
{{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "risk_score": 0-100,
  "risk_factors": ["factor1", "factor2"],
  "detailed_assessment": "explanation",
  "recommended_actions": ["action1", "action2"],
  "confidence": 0.0-1.0
}}"""

        response = await self.model_manager.query_claude(risk_prompt)
        
        try:
            assessment = json.loads(response)
            return assessment
        except:
            return {
                "risk_level": "UNKNOWN",
                "risk_score": 0,
                "risk_factors": [],
                "detailed_assessment": "Unable to assess",
                "recommended_actions": [],
                "confidence": 0.0
            }


# ============================================================================
# CONNECTION MAPPER
# ============================================================================

class ConnectionMapper:
    """Maps and analyzes entity relationships"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.graph = nx.DiGraph()
    
    async def extract_connections(self, entity_name: str, facts: List[Fact]) -> List[Dict[str, Any]]:
        """Extract entity connections from facts"""
        
        facts_text = "\n".join([f"- {f.content}" for f in facts])
        
        connection_prompt = f"""Extract all entity connections (people, organizations, locations) for: {entity_name}

Facts:
{facts_text}

Return JSON array of connections:
[
  {{
    "connected_entity": "name",
    "entity_type": "person|organization|location",
    "relationship": "description",
    "strength": 0.0-1.0,
    "context": "explanation",
    "temporal": "time period if known"
  }}
]"""

        response = await self.model_manager.query_claude(connection_prompt)
        
        try:
            connections = json.loads(response)
            
            # Build graph
            for conn in connections:
                self.graph.add_edge(
                    entity_name,
                    conn["connected_entity"],
                    relationship=conn["relationship"],
                    strength=conn["strength"],
                    context=conn["context"]
                )
            
            return connections
        except:
            return []
    
    def analyze_network(self, entity_name: str) -> Dict[str, Any]:
        """Analyze network structure and identify patterns"""
        if entity_name not in self.graph:
            return {}
        
        try:
            # Network metrics
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness = nx.betweenness_centrality(self.graph)
            
            # Find influential connections
            neighbors = list(self.graph.neighbors(entity_name))
            influential = sorted(
                neighbors,
                key=lambda n: degree_centrality.get(n, 0),
                reverse=True
            )[:5]
            
            # Identify clusters
            try:
                communities = list(nx.community.greedy_modularity_communities(
                    self.graph.to_undirected()
                ))
                entity_community = [c for c in communities if entity_name in c]
            except:
                entity_community = []
            
            return {
                "direct_connections": len(neighbors),
                "centrality_score": degree_centrality.get(entity_name, 0),
                "betweenness_score": betweenness.get(entity_name, 0),
                "influential_connections": influential,
                "community_size": len(entity_community[0]) if entity_community else 0,
                "network_depth": nx.eccentricity(self.graph, entity_name) if nx.is_connected(self.graph.to_undirected()) else 0
            }
        except Exception as e:
            logger.error(f"Network analysis error: {e}")
            return {}


# ============================================================================
# QUERY ORCHESTRATOR
# ============================================================================

class QueryOrchestrator:
    """Generates and manages search query progression"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.query_history = []
    
    async def generate_next_queries(self, state: AgentState) -> List[SearchQuery]:
        """Generate contextual follow-up queries"""
        
        # Analyze what we know and what we need
        facts_summary = "\n".join([
            f"- {f.content} (category: {f.category}, confidence: {f.confidence})"
            for f in state["discovered_facts"][-20:]  # Last 20 facts
        ])
        
        executed_queries = [q.query for q in state["executed_queries"]]
        
        strategy_prompt = f"""You are a strategic intelligence analyst. Generate the next set of search queries.

Target: {state['target_name']}
Current Phase: {state['current_phase']}

Recent Discoveries:
{facts_summary}

Previously Executed Queries:
{chr(10).join([f"- {q}" for q in executed_queries[-10:]])}

Phase Guidelines:
- Phase 1 (Discovery): Basic biographical, current positions, public presence
- Phase 2 (Professional): Career history, company affiliations, board memberships
- Phase 3 (Network): Business partners, family, associates
- Phase 4 (Risk): Legal issues, financial irregularities, reputation events
- Phase 5 (Hidden): Indirect relationships, historical patterns, name variations

Generate 3-5 specific, strategic queries that:
1. Build on previous findings
2. Explore gaps in knowledge
3. Look for contradictions or red flags
4. Dig deeper into concerning areas
5. Use specific names, companies, dates from discoveries

Return JSON:
{{
  "queries": [
    {{
      "query": "specific search query",
      "reasoning": "why this query",
      "expected_findings": "what we hope to discover",
      "depth": "surface|medium|deep|maximum"
    }}
  ],
  "phase_complete": boolean
}}"""

        response = await self.model_manager.query_claude(strategy_prompt)
        
        try:
            result = json.loads(response)
            queries = []
            
            for q in result.get("queries", []):
                depth_map = {
                    "surface": SearchDepth.SURFACE,
                    "medium": SearchDepth.MEDIUM,
                    "deep": SearchDepth.DEEP,
                    "maximum": SearchDepth.MAXIMUM
                }
                
                query = SearchQuery(
                    query=q["query"],
                    phase=state["current_phase"],
                    depth=depth_map.get(q.get("depth", "medium"), SearchDepth.MEDIUM),
                    context={
                        "reasoning": q.get("reasoning"),
                        "expected": q.get("expected_findings")
                    }
                )
                queries.append(query)
            
            return queries
        except Exception as e:
            logger.error(f"Query generation error: {e}")
            return []


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates comprehensive assessment reports"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    async def generate_report(self, state: AgentState) -> str:
        """Generate final comprehensive report"""
        
        # Organize all collected data
        entity = state["entities"].get(state["target_name"], Entity(
            name=state["target_name"],
            entity_type="unknown"
        ))
        
        facts_by_category = defaultdict(list)
        for fact in state["discovered_facts"]:
            facts_by_category[fact.category].append(fact)
        
        report_prompt = f"""Generate a comprehensive intelligence assessment report.

TARGET: {state['target_name']}
TOTAL FACTS DISCOVERED: {len(state['discovered_facts'])}
SEARCH QUERIES EXECUTED: {len(state['executed_queries'])}
RISK ASSESSMENT: {json.dumps(state.get('risk_assessment', {}), indent=2)}

FACTS BY CATEGORY:
{chr(10).join([
    f"{cat.upper()}: {len(facts)} facts" 
    for cat, facts in facts_by_category.items()
])}

CONNECTIONS IDENTIFIED: {len(state['connections_graph'].get(state['target_name'], []))}

Generate a professional intelligence report with:

## EXECUTIVE SUMMARY
- Key findings overview
- Risk level and score
- Critical concerns

## BIOGRAPHICAL INFORMATION
- Verified personal details
- Professional background
- Current positions

## PROFESSIONAL HISTORY
- Career trajectory
- Company affiliations
- Board positions

## NETWORK ANALYSIS
- Key relationships
- Business connections
- Family ties
- Concerning associations

## RISK ASSESSMENT
- Identified risk factors
- Legal/regulatory issues
- Financial concerns
- Reputational risks

## HIDDEN CONNECTIONS
- Indirect relationships
- Historical patterns
- Name variations or aliases
- Obscure affiliations

## VERIFICATION STATUS
- High confidence facts
- Medium confidence facts
- Unverified claims requiring further investigation

## RECOMMENDATIONS
- Areas requiring deeper investigation
- Monitoring suggestions
- Risk mitigation strategies

## METHODOLOGY
- Search phases completed
- Data sources consulted
- Models used
- Limitations

Make it professional, detailed, and actionable."""

        # Use consensus from multiple models for critical report
        responses = await self.model_manager.consensus_query(report_prompt)
        
        # Combine insights
        final_report = f"""# DEEP RESEARCH INTELLIGENCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target: {state['target_name']}

{'='*80}

## PRIMARY ANALYSIS (Claude)
{responses.get('claude', 'N/A')}

{'='*80}

## SECONDARY ANALYSIS (GPT)
{responses.get('openai', 'N/A')}

{'='*80}

## METADATA
- Total Facts: {len(state['discovered_facts'])}
- Queries Executed: {len(state['executed_queries'])}
- Phases Completed: {state['current_phase']}
- Verified Facts: {sum(1 for f in state['discovered_facts'] if f.verified)}
- High Confidence Facts: {sum(1 for f in state['discovered_facts'] if f.confidence > 0.8)}

## FACT CATEGORIES
{chr(10).join([f"- {cat}: {len(facts)}" for cat, facts in facts_by_category.items()])}
"""
        
        return final_report


# ============================================================================
# MAIN RESEARCH AGENT
# ============================================================================

class DeepResearchAgent:
    """Main orchestrator for the research agent"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.search_engine = SearchEngine()
        self.fact_extractor = FactExtractor(self.model_manager)
        self.risk_analyzer = RiskAnalyzer(self.model_manager)
        self.connection_mapper = ConnectionMapper(self.model_manager)
        self.query_orchestrator = QueryOrchestrator(self.model_manager)
        self.report_generator = ReportGenerator(self.model_manager)
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("initialize", self.initialize_node)
        workflow.add_node("generate_queries", self.generate_queries_node)
        workflow.add_node("execute_search", self.execute_search_node)
        workflow.add_node("extract_facts", self.extract_facts_node)
        workflow.add_node("analyze_risks", self.analyze_risks_node)
        workflow.add_node("map_connections", self.map_connections_node)
        workflow.add_node("generate_report", self.generate_report_node)
        
        # Define edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "generate_queries")
        workflow.add_edge("generate_queries", "execute_search")
        workflow.add_edge("execute_search", "extract_facts")
        workflow.add_edge("extract_facts", "analyze_risks")
        workflow.add_edge("analyze_risks", "map_connections")
        workflow.add_conditional_edges(
            "map_connections",
            self.should_continue,
            {
                "continue": "generate_queries",
                "finish": "generate_report"
            }
        )
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    async def initialize_node(self, state: AgentState) -> AgentState:
        """Initialize the research process"""
        logger.info(f"Initializing research for: {state['target_name']}")
        state["current_phase"] = 1
        state["executed_queries"] = []
        state["discovered_facts"] = []
        state["entities"] = {}
        state["connections_graph"] = {}
        state["risk_assessment"] = {}
        state["next_queries"] = []
        state["error_log"] = []
        return state
    
    async def generate_queries_node(self, state: AgentState) -> AgentState:
        """Generate next set of queries"""
        logger.info(f"Generating queries for phase {state['current_phase']}")
        
        if not state["next_queries"]:
            queries = await self.query_orchestrator.generate_next_queries(state)
            state["next_queries"] = queries
        
        return state
    
    async def execute_search_node(self, state: AgentState) -> AgentState:
        """Execute search queries"""
        if not state["next_queries"]:
            return state
        
        query = state["next_queries"].pop(0)
        logger.info(f"Executing search: {query.query}")
        
        try:
            results = await self.search_engine.multi_source_search(
                query.query,
                query.depth
            )
            query.context["results"] = results
            state["executed_queries"].append(query)
        except Exception as e:
            logger.error(f"Search error: {e}")
            state["error_log"].append(f"Search failed: {query.query} - {str(e)}")
        
        return state
    
    async def extract_facts_node(self, state: AgentState) -> AgentState:
        """Extract facts from search results"""
        if not state["executed_queries"]:
            return state
        
        last_query = state["executed_queries"][-1]
        results = last_query.context.get("results", [])
        
        logger.info(f"Extracting facts from {len(results)} results")
        
        try:
            facts = await self.fact_extractor.extract_facts(
                last_query.query,
                results,
                last_query.context
            )
            
            # Verify facts
            for fact in facts:
                fact.verified = await self.fact_extractor.verify_fact(
                    fact,
                    state["discovered_facts"]
                )
            
            state["discovered_facts"].extend(facts)
            logger.info(f"Extracted {len(facts)} facts ({sum(1 for f in facts if f.verified)} verified)")
        except Exception as e:
            logger.error(f"Fact extraction error: {e}")
            state["error_log"].append(f"Fact extraction failed: {str(e)}")
        
        return state
    
    async def analyze_risks_node(self, state: AgentState) -> AgentState:
        """Analyze risks"""
        entity_name = state["target_name"]
        
        if entity_name not in state["entities"]:
            state["entities"][entity_name] = Entity(
                name=entity_name,
                entity_type="person",
                facts=state["discovered_facts"]
            )
        
        entity = state["entities"][entity_name]
        entity.facts = state["discovered_facts"]
        
        logger.info("Analyzing risks")
        
        try:
            assessment = await self.risk_analyzer.analyze_risks(
                entity,
                state["discovered_facts"]
            )
            state["risk_assessment"] = assessment
            entity.risk_score = assessment.get("risk_score", 0)
            entity.risk_factors = assessment.get("risk_factors", [])
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            state["error_log"].append(f"Risk analysis failed: {str(e)}")
        
        return state
    
    async def map_connections_node(self, state: AgentState) -> AgentState:
        """Map entity connections"""
        entity_name = state["target_name"]
        
        logger.info("Mapping connections")
        
        try:
            connections = await self.connection_mapper.extract_connections(
                entity_name,
                state["discovered_facts"]
            )
            state["connections_graph"][entity_name] = connections
        except Exception as e:
            logger.error(f"Connection mapping error: {e}")
            state["error_log"].append(f"Connection mapping failed: {str(e)}")
        
        return state
    
    def should_continue(self, state: AgentState) -> str:
        """Decide whether to continue searching or finish"""
        max_phases = 5
        min_facts = 20
        
        # Continue if not enough facts and haven't completed all phases
        if (state["current_phase"] < max_phases and 
            len(state["discovered_facts"]) < min_facts * state["current_phase"]):
            state["current_phase"] += 1
            return "continue"
        
        # Continue if there are more queries in the current phase
        if state["next_queries"]:
            return "continue"
        
        return "finish"
    
    async def generate_report_node(self, state: AgentState) -> AgentState:
        """Generate final report"""
        logger.info("Generating final report")
        
        try:
            report = await self.report_generator.generate_report(state)
            state["final_report"] = report
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            state["error_log"].append(f"Report generation failed: {str(e)}")
            state["final_report"] = "Error generating report"
        
        return state
    
    async def research(self, target_name: str) -> Dict[str, Any]:
        """Main research method"""
        initial_state: AgentState = {
            "target_name": target_name,
            "current_phase": 0,
            "executed_queries": [],
            "discovered_facts": [],
            "entities": {},
            "connections_graph": {},
            "risk_assessment": {},
            "next_queries": [],
            "final_report": None,
            "error_log": []
        }
        
        # Execute the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "target": target_name,
            "report": final_state["final_report"],
            "total_facts": len(final_state["discovered_facts"]),
            "queries_executed": len(final_state["executed_queries"]),
            "risk_assessment": final_state["risk_assessment"],
            "connections": final_state["connections_graph"],
            "errors": final_state["error_log"]
        }


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

@dataclass
class TestPersona:
    """Test persona with hidden facts"""
    name: str
    description: str
    hidden_facts: List[Dict[str, Any]]
    expected_connections: List[str]
    expected_risk_level: str


class EvaluationFramework:
    """Evaluates agent performance"""
    
    def __init__(self):
        self.test_personas = self._create_test_personas()
    
    def _create_test_personas(self) -> List[TestPersona]:
        """Create test personas with hidden facts"""
        return [
            TestPersona(
                name="Sarah Chen",
                description="Tech Executive at hypothetical VisionTech Corp",
                hidden_facts=[
                    {"fact": "Changed name from Sarah Zhang in 2010", "difficulty": "hard"},
                    {"fact": "Silent partner in failed CloudStart startup 2015", "difficulty": "very_hard"},
                    {"fact": "Cousin works at SEC", "difficulty": "extreme"},
                    {"fact": "Board member of offshore consulting firm", "difficulty": "extreme"},
                    {"fact": "Published AI ethics paper under pseudonym 'S. Chen-Wu'", "difficulty": "hard"}
                ],
                expected_connections=["VisionTech Corp", "CloudStart", "SEC", "family members"],
                expected_risk_level="MEDIUM"
            ),
            TestPersona(
                name="Michael Roberts",
                description="Financial Consultant at hypothetical Apex Advisory",
                hidden_facts=[
                    {"fact": "Previously employed at sanctioned investment firm", "difficulty": "medium"},
                    {"fact": "Shared business address with fraud investigation subject", "difficulty": "hard"},
                    {"fact": "Indirect investment in competitor through family trust", "difficulty": "very_hard"},
                    {"fact": "Wife owns undisclosed consulting business", "difficulty": "hard"},
                    {"fact": "MBA from two different schools under name variation", "difficulty": "medium"}
                ],
                expected_connections=["Apex Advisory", "family trust", "wife's business"],
                expected_risk_level="HIGH"
            )
        ]
    
    def evaluate_results(self, persona: TestPersona, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent performance on a test persona"""
        discovered_facts = results.get("total_facts", 0)
        
        # Check if hidden facts were found
        hidden_facts_found = 0
        report = results.get("report", "").lower()
        
        for hidden_fact in persona.hidden_facts:
            # Simple keyword matching (in production, use more sophisticated matching)
            fact_keywords = hidden_fact["fact"].lower().split()
            if any(keyword in report for keyword in fact_keywords):
                hidden_facts_found += 1
        
        discovery_rate = hidden_facts_found / len(persona.hidden_facts) if persona.hidden_facts else 0
        
        return {
            "persona": persona.name,
            "total_facts_discovered": discovered_facts,
            "hidden_facts_found": hidden_facts_found,
            "hidden_facts_total": len(persona.hidden_facts),
            "discovery_rate": discovery_rate,
            "queries_executed": results.get("queries_executed", 0),
            "risk_assessment_match": results.get("risk_assessment", {}).get("risk_level") == persona.expected_risk_level,
            "score": (discovery_rate * 0.6 + (1 if results.get("risk_assessment", {}).get("risk_level") == persona.expected_risk_level else 0) * 0.4)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function"""
    
    # Initialize agent
    agent = DeepResearchAgent()
    evaluator = EvaluationFramework()
    
    # Run evaluation
    print("=" * 80)
    print("DEEP RESEARCH AI AGENT - EVALUATION")
    print("=" * 80)
    
    results = []
    
    for persona in evaluator.test_personas:
        print(f"\nResearching: {persona.name}")
        print(f"Description: {persona.description}")
        print(f"Hidden facts to discover: {len(persona.hidden_facts)}")
        print("-" * 80)
        
        try:
            # Execute research
            research_results = await agent.research(persona.name)
            
            # Evaluate
            evaluation = evaluator.evaluate_results(persona, research_results)
            results.append(evaluation)
            
            # Print results
            print(f"\nResults for {persona.name}:")
            print(f"  Total facts discovered: {evaluation['total_facts_discovered']}")
            print(f"  Hidden facts found: {evaluation['hidden_facts_found']}/{evaluation['hidden_facts_total']}")
            print(f"  Discovery rate: {evaluation['discovery_rate']:.2%}")
            print(f"  Overall score: {evaluation['score']:.2f}")
            
            # Save report
            with open(f"report_{persona.name.replace(' ', '_')}.md", "w") as f:
                f.write(research_results["report"])
            print(f"  Report saved to: report_{persona.name.replace(' ', '_')}.md")
            
        except Exception as e:
            print(f"Error researching {persona.name}: {e}")
            logger.exception(e)
    
    # Overall evaluation
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        print("\n" + "=" * 80)
        print(f"OVERALL AGENT PERFORMANCE: {avg_score:.2f}/1.00")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())