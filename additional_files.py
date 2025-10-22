"""
Additional Implementation Files for Deep Research Agent
Contains: CLI, Testing, Utilities, and Configuration
"""

# ============================================================================
# FILE: cli.py - Command Line Interface
# ============================================================================

import argparse
import asyncio
import sys
from pathlib import Path
from deep_research_agent import DeepResearchAgent, EvaluationFramework

def create_parser():
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Deep Research AI Agent - Autonomous Intelligence Gathering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single target
  python cli.py --target "John Doe" --output report.md
  
  # Batch processing
  python cli.py --batch targets.txt --output-dir ./reports/
  
  # Evaluation mode
  python cli.py --evaluate --personas test_personas.json
  
  # Custom configuration
  python cli.py --target "Jane Smith" --phases 7 --depth maximum
        """
    )
    
    # Target specification
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target", "-t",
        help="Single target name to research"
    )
    target_group.add_argument(
        "--batch", "-b",
        help="File containing list of targets (one per line)"
    )
    target_group.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Run evaluation mode with test personas"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file for report (default: report_{target}.md)"
    )
    parser.add_argument(
        "--output-dir",
        default="./reports",
        help="Directory for batch reports (default: ./reports)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "html", "pdf"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    
    # Research configuration
    parser.add_argument(
        "--phases",
        type=int,
        default=5,
        help="Number of research phases (default: 5)"
    )
    parser.add_argument(
        "--depth",
        choices=["surface", "medium", "deep", "maximum"],
        default="deep",
        help="Search depth (default: deep)"
    )
    parser.add_argument(
        "--focus",
        help="Focus areas (comma-separated): biographical,professional,financial,legal"
    )
    parser.add_argument(
        "--queries-per-phase",
        type=int,
        default=5,
        help="Max queries per phase (default: 5)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--personas",
        help="JSON file with test personas for evaluation"
    )
    
    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with detailed logging"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    return parser


async def research_single(args, agent):
    """Research a single target"""
    print(f"\n{'='*80}")
    print(f"Researching: {args.target}")
    print(f"{'='*80}\n")
    
    results = await agent.research(args.target)
    
    # Determine output file
    output_file = args.output or f"report_{args.target.replace(' ', '_')}.md"
    
    # Save report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(results["report"])
    
    print(f"\n{'='*80}")
    print(f"Research Complete!")
    print(f"{'='*80}")
    print(f"Total Facts: {results['total_facts']}")
    print(f"Queries Executed: {results['queries_executed']}")
    print(f"Risk Level: {results['risk_assessment'].get('risk_level', 'N/A')}")
    print(f"Report saved to: {output_file}")
    
    if results.get('errors'):
        print(f"\nWarnings/Errors: {len(results['errors'])}")
        for error in results['errors'][:3]:
            print(f"  - {error}")


async def research_batch(args, agent):
    """Research multiple targets from file"""
    targets_file = Path(args.batch)
    if not targets_file.exists():
        print(f"Error: Targets file not found: {targets_file}")
        sys.exit(1)
    
    targets = [line.strip() for line in targets_file.read_text().splitlines() if line.strip()]
    
    print(f"\n{'='*80}")
    print(f"Batch Research: {len(targets)} targets")
    print(f"{'='*80}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] Researching: {target}")
        
        try:
            results = await agent.research(target)
            
            output_file = output_dir / f"report_{target.replace(' ', '_')}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(results["report"])
            
            print(f"  ✓ Complete: {results['total_facts']} facts, saved to {output_file}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"Batch Complete!")
    print(f"Reports saved to: {output_dir}")
    print(f"{'='*80}")


async def run_evaluation(args):
    """Run evaluation mode"""
    print(f"\n{'='*80}")
    print("EVALUATION MODE")
    print(f"{'='*80}\n")
    
    evaluator = EvaluationFramework()
    agent = DeepResearchAgent()
    
    results = []
    for persona in evaluator.test_personas:
        print(f"\nEvaluating: {persona.name}")
        print(f"Hidden facts: {len(persona.hidden_facts)}")
        
        research_results = await agent.research(persona.name)
        evaluation = evaluator.evaluate_results(persona, research_results)
        results.append(evaluation)
        
        print(f"  Discovery rate: {evaluation['discovery_rate']:.2%}")
        print(f"  Score: {evaluation['score']:.2f}")
    
    avg_score = sum(r['score'] for r in results) / len(results)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"Overall Score: {avg_score:.2%}")
    print(f"{'='*80}")


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    import logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    try:
        if args.evaluate:
            await run_evaluation(args)
        else:
            # Initialize agent with configuration
            agent = DeepResearchAgent()
            
            if args.phases:
                agent.config.max_phases = args.phases
            if args.queries_per_phase:
                agent.config.queries_per_phase = args.queries_per_phase
            if args.no_cache:
                agent.search_engine.disable_cache()
            
            if args.target:
                await research_single(args, agent)
            elif args.batch:
                await research_batch(args, agent)
    
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================================
# FILE: test_agent.py - Comprehensive Test Suite
# ============================================================================

import pytest
import asyncio
from deep_research_agent import (
    DeepResearchAgent, FactExtractor, RiskAnalyzer,
    ConnectionMapper, QueryOrchestrator, ModelManager,
    Fact, Entity, SearchQuery, SearchDepth
)

@pytest.fixture
def model_manager():
    """Create model manager for tests"""
    return ModelManager()

@pytest.fixture
def agent():
    """Create agent for tests"""
    return DeepResearchAgent()


class TestFactExtractor:
    """Test fact extraction functionality"""
    
    @pytest.mark.asyncio
    async def test_extract_facts_basic(self, model_manager):
        extractor = FactExtractor(model_manager)
        
        search_results = [
            {
                "url": "https://example.com/bio",
                "title": "John Doe Biography",
                "content": "John Doe was born in 1980 and became CEO in 2020."
            }
        ]
        
        facts = await extractor.extract_facts(
            "John Doe biography",
            search_results,
            {}
        )
        
        assert len(facts) > 0
        assert all(isinstance(f, Fact) for f in facts)
        assert all(0 <= f.confidence <= 1 for f in facts)
    
    @pytest.mark.asyncio
    async def test_fact_verification(self, model_manager):
        extractor = FactExtractor(model_manager)
        
        fact1 = Fact(
            content="John Doe is CEO of TechCorp",
            source="source1.com",
            confidence=0.8,
            timestamp=datetime.now(),
            category="professional"
        )
        
        similar_facts = [
            Fact(
                content="John Doe leads TechCorp as Chief Executive",
                source="source2.com",
                confidence=0.85,
                timestamp=datetime.now(),
                category="professional"
            ),
            Fact(
                content="TechCorp CEO John Doe",
                source="source3.com",
                confidence=0.9,
                timestamp=datetime.now(),
                category="professional"
            )
        ]
        
        verified = await extractor.verify_fact(fact1, similar_facts)
        assert verified is True


class TestRiskAnalyzer:
    """Test risk analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_risk_patterns(self, model_manager):
        analyzer = RiskAnalyzer(model_manager)
        
        entity = Entity(
            name="Test Person",
            entity_type="person",
            facts=[
                Fact(
                    content="Subject of SEC investigation in 2020",
                    source="sec.gov",
                    confidence=0.95,
                    timestamp=datetime.now(),
                    category="legal"
                ),
                Fact(
                    content="Resigned amid scandal",
                    source="news.com",
                    confidence=0.85,
                    timestamp=datetime.now(),
                    category="reputational"
                )
            ]
        )
        
        assessment = await analyzer.analyze_risks(entity, entity.facts)
        
        assert assessment["risk_score"] > 50
        assert assessment["risk_level"] in ["HIGH", "CRITICAL"]
        assert len(assessment["risk_factors"]) > 0


class TestConnectionMapper:
    """Test connection mapping functionality"""
    
    @pytest.mark.asyncio
    async def test_extract_connections(self, model_manager):
        mapper = ConnectionMapper(model_manager)
        
        facts = [
            Fact(
                content="John Doe is partner at ABC Law with Jane Smith",
                source="example.com",
                confidence=0.9,
                timestamp=datetime.now(),
                category="connection"
            ),
            Fact(
                content="Board member of XYZ Corp alongside Michael Brown",
                source="example.com",
                confidence=0.85,
                timestamp=datetime.now(),
                category="professional"
            )
        ]
        
        connections = await mapper.extract_connections("John Doe", facts)
        
        assert len(connections) > 0
        assert any(c["connected_entity"] == "Jane Smith" for c in connections)
        assert any(c["connected_entity"] == "Michael Brown" for c in connections)
    
    def test_network_analysis(self, model_manager):
        mapper = ConnectionMapper(model_manager)
        
        # Build test network
        mapper.graph.add_edge("John Doe", "Jane Smith", relationship="partner")
        mapper.graph.add_edge("John Doe", "XYZ Corp", relationship="board_member")
        mapper.graph.add_edge("Jane Smith", "ABC Law", relationship="employee")
        
        analysis = mapper.analyze_network("John Doe")
        
        assert analysis["direct_connections"] == 2
        assert analysis["centrality_score"] > 0


class TestQueryOrchestrator:
    """Test query generation and orchestration"""
    
    @pytest.mark.asyncio
    async def test_generate_next_queries(self, model_manager):
        orchestrator = QueryOrchestrator(model_manager)
        
        state = {
            "target_name": "John Doe",
            "current_phase": 2,
            "discovered_facts": [
                Fact(
                    content="CEO of TechCorp",
                    source="example.com",
                    confidence=0.9,
                    timestamp=datetime.now(),
                    category="professional"
                )
            ],
            "executed_queries": [
                SearchQuery(
                    query="John Doe biography",
                    phase=1,
                    depth=SearchDepth.SURFACE
                )
            ]
        }
        
        queries = await orchestrator.generate_next_queries(state)
        
        assert len(queries) > 0
        assert all(isinstance(q, SearchQuery) for q in queries)
        assert all(q.phase == 2 for q in queries)


class TestIntegration:
    """Integration tests for full agent workflow"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_research_flow(self, agent):
        """Test complete research workflow"""
        # Use a fictional name to avoid real data issues
        results = await agent.research("Test Person XYZ")
        
        assert results is not None
        assert "report" in results
        assert "total_facts" in results
        assert results["total_facts"] >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test agent handles errors gracefully"""
        # Intentionally cause an error with invalid input
        agent.search_engine.tavily_key = "invalid_key"
        
        results = await agent.research("Test Person")
        
        # Should complete without crashing
        assert results is not None
        assert len(results.get("errors", [])) > 0


# ============================================================================
# FILE: config.py - Configuration Management
# ============================================================================

from dataclasses import dataclass
from typing import Dict, Any
import os
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """AI Model configuration"""
    primary_model: str = "claude-sonnet-4-5-20250929"
    deep_analysis_model: str = "claude-opus-4-20250514"
    alternative_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: float = 60.0


@dataclass
class SearchConfig:
    """Search engine configuration"""
    max_results_per_query: int = 15
    cache_enabled: bool = True
    cache_ttl: int = 3600
    concurrent_searches: int = 3
    search_timeout: float = 30.0


@dataclass
class AgentConfig:
    """Agent behavior configuration"""
    max_phases: int = 5
    min_facts_per_phase: int = 20
    queries_per_phase: int = 5
    verification_threshold: float = 0.7
    risk_score_threshold: float = 60.0
    
    # Rate limiting
    requests_per_minute: int = 50
    concurrent_requests: int = 5
    
    # Output
    output_format: str = "markdown"
    save_intermediate: bool = True
    debug_mode: bool = False


class ConfigManager:
    """Manages configuration from multiple sources"""
    
    def __init__(self, config_file: str = None):
        self.model_config = ModelConfig()
        self.search_config = SearchConfig()
        self.agent_config = AgentConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
    
    def load_from_file(self, file_path: str):
        """Load configuration from JSON file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(path) as f:
            config_data = json.load(f)
        
        # Update configurations
        if "model" in config_data:
            for key, value in config_data["model"].items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        if "search" in config_data:
            for key, value in config_data["search"].items():
                if hasattr(self.search_config, key):
                    setattr(self.search_config, key, value)
        
        if "agent" in config_data:
            for key, value in config_data["agent"].items():
                if hasattr(self.agent_config, key):
                    setattr(self.agent_config, key, value)
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Model config
        if os.getenv("PRIMARY_MODEL"):
            self.model_config.primary_model = os.getenv("PRIMARY_MODEL")
        if os.getenv("MODEL_TEMPERATURE"):
            self.model_config.temperature = float(os.getenv("MODEL_TEMPERATURE"))
        
        # Search config
        if os.getenv("MAX_SEARCH_RESULTS"):
            self.search_config.max_results_per_query = int(os.getenv("MAX_SEARCH_RESULTS"))
        if os.getenv("CACHE_TTL"):
            self.search_config.cache_ttl = int(os.getenv("CACHE_TTL"))
        
        # Agent config
        if os.getenv("MAX_PHASES"):
            self.agent_config.max_phases = int(os.getenv("MAX_PHASES"))
        if os.getenv("DEBUG_MODE"):
            self.agent_config.debug_mode = os.getenv("DEBUG_MODE").lower() == "true"
    
    def save_to_file(self, file_path: str):
        """Save current configuration to file"""
        config_data = {
            "model": {
                "primary_model": self.model_config.primary_model,
                "deep_analysis_model": self.model_config.deep_analysis_model,
                "alternative_model": self.model_config.alternative_model,
                "temperature": self.model_config.temperature,
                "max_tokens": self.model_config.max_tokens,
                "timeout": self.model_config.timeout
            },
            "search": {
                "max_results_per_query": self.search_config.max_results_per_query,
                "cache_enabled": self.search_config.cache_enabled,
                "cache_ttl": self.search_config.cache_ttl,
                "concurrent_searches": self.search_config.concurrent_searches,
                "search_timeout": self.search_config.search_timeout
            },
            "agent": {
                "max_phases": self.agent_config.max_phases,
                "min_facts_per_phase": self.agent_config.min_facts_per_phase,
                "queries_per_phase": self.agent_config.queries_per_phase,
                "verification_threshold": self.agent_config.verification_threshold,
                "risk_score_threshold": self.agent_config.risk_score_threshold,
                "requests_per_minute": self.agent_config.requests_per_minute,
                "concurrent_requests": self.agent_config.concurrent_requests,
                "output_format": self.agent_config.output_format,
                "save_intermediate": self.agent_config.save_intermediate,
                "debug_mode": self.agent_config.debug_mode
            }
        }
        
        with open(file_path, "w") as f:
            json.dump(config_data, f, indent=2)


# ============================================================================
# FILE: utils.py - Utility Functions
# ============================================================================

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import hashlib
import re
from urllib.parse import urlparse


def sanitize_filename(name: str) -> str:
    """Convert name to valid filename"""
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Limit length
    if len(name) > 200:
        name = name[:200]
    return name


def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except:
        return ""


def calculate_hash(text: str) -> str:
    """Calculate SHA256 hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()


def is_recent(date_str: str, days: int = 365) -> bool:
    """Check if date is within specified days"""
    try:
        # Try multiple date formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%B %d, %Y"]:
            try:
                date = datetime.strptime(date_str, fmt)
                return (datetime.now() - date).days <= days
            except:
                continue
        return False
    except:
        return False


def deduplicate_facts(facts: List[Fact]) -> List[Fact]:
    """Remove duplicate facts based on content similarity"""
    unique_facts = []
    seen_hashes = set()
    
    for fact in facts:
        # Create hash of normalized content
        normalized = fact.content.lower().strip()
        content_hash = calculate_hash(normalized)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_facts.append(fact)
    
    return unique_facts


def merge_similar_facts(facts: List[Fact], threshold: float = 0.8) -> List[Fact]:
    """Merge facts with similar content"""
    from difflib import SequenceMatcher
    
    merged = []
    processed = set()
    
    for i, fact1 in enumerate(facts):
        if i in processed:
            continue
        
        similar_facts = [fact1]
        for j, fact2 in enumerate(facts[i+1:], start=i+1):
            if j in processed:
                continue
            
            # Calculate similarity
            similarity = SequenceMatcher(
                None,
                fact1.content.lower(),
                fact2.content.lower()
            ).ratio()
            
            if similarity >= threshold:
                similar_facts.append(fact2)
                processed.add(j)
        
        # Merge facts
        if len(similar_facts) > 1:
            # Use fact with highest confidence
            best_fact = max(similar_facts, key=lambda f: f.confidence)
            best_fact.cross_references = [f.source for f in similar_facts]
            best_fact.verified = True
            merged.append(best_fact)
        else:
            merged.append(fact1)
        
        processed.add(i)
    
    return merged


def format_timeline(facts: List[Fact]) -> str:
    """Create timeline visualization of facts"""
    # Extract facts with temporal information
    temporal_facts = [
        f for f in facts
        if any(str(year) in f.content for year in range(1950, 2026))
    ]
    
    # Sort by extracted year
    def extract_year(fact: Fact) -> int:
        years = re.findall(r'\b(19|20)\d{2}\b', fact.content)
        return int(years[0]) if years else 9999
    
    temporal_facts.sort(key=extract_year)
    
    # Format as timeline
    timeline = "## Timeline\n\n"
    for fact in temporal_facts:
        year = extract_year(fact)
        if year != 9999:
            timeline += f"**{year}**: {fact.content}\n\n"
    
    return timeline


def calculate_confidence_aggregate(facts: List[Fact]) -> Dict[str, float]:
    """Calculate aggregate confidence metrics"""
    if not facts:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    
    confidences = [f.confidence for f in facts]
    
    return {
        "avg": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "median": sorted(confidences)[len(confidences) // 2]
    }


def generate_summary_stats(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics from agent state"""
    facts = state.get("discovered_facts", [])
    queries = state.get("executed_queries", [])
    
    # Facts by category
    from collections import Counter
    categories = Counter(f.category for f in facts)
    
    # Verification stats
    verified = sum(1 for f in facts if f.verified)
    high_confidence = sum(1 for f in facts if f.confidence > 0.8)
    
    # Query stats
    queries_by_phase = Counter(q.phase for q in queries)
    
    return {
        "total_facts": len(facts),
        "verified_facts": verified,
        "high_confidence_facts": high_confidence,
        "facts_by_category": dict(categories),
        "total_queries": len(queries),
        "queries_by_phase": dict(queries_by_phase),
        "avg_confidence": calculate_confidence_aggregate(facts)["avg"],
        "phases_completed": state.get("current_phase", 0)
    }


class ProgressTracker:
    """Track and display research progress"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.events = []
    
    def log_event(self, event_type: str, message: str):
        """Log a progress event"""
        self.events.append({
            "timestamp": datetime.now(),
            "type": event_type,
            "message": message
        })
    
    def get_duration(self) -> timedelta:
        """Get total duration"""
        return datetime.now() - self.start_time
    
    def get_summary(self) -> str:
        """Get progress summary"""
        duration = self.get_duration()
        event_counts = {}
        
        for event in self.events:
            event_type = event["type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        summary = f"Duration: {duration}\n"
        summary += "Events:\n"
        for event_type, count in event_counts.items():
            summary += f"  {event_type}: {count}\n"
        
        return summary


# ============================================================================
# FILE: export.py - Export Utilities
# ============================================================================

import json
import csv
from typing import List, Dict
from pathlib import Path


class ReportExporter:
    """Export reports in various formats"""
    
    @staticmethod
    def export_json(results: Dict[str, Any], output_file: str):
        """Export results as JSON"""
        # Convert non-serializable objects
        serializable_results = {
            "target": results["target"],
            "report": results["report"],
            "total_facts": results["total_facts"],
            "queries_executed": results["queries_executed"],
            "risk_assessment": results["risk_assessment"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
    
    @staticmethod
    def export_csv_facts(facts: List[Fact], output_file: str):
        """Export facts as CSV"""
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Content", "Source", "Confidence", 
                "Category", "Verified", "Timestamp"
            ])
            
            for fact in facts:
                writer.writerow([
                    fact.content,
                    fact.source,
                    fact.confidence,
                    fact.category,
                    fact.verified,
                    fact.timestamp.isoformat()
                ])
    
    @staticmethod
    def export_html(results: Dict[str, Any], output_file: str):
        """Export report as HTML"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Research Report - {target}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .risk-low {{ background: #d4edda; color: #155724; }}
        .risk-medium {{ background: #fff3cd; color: #856404; }}
        .risk-high {{ background: #f8d7da; color: #721c24; }}
        .risk-critical {{ background: #d73a4a; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Research Report: {target}</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Summary Metrics</h2>
        <div class="metric">
            <div class="metric-value">{total_facts}</div>
            <div class="metric-label">Facts Discovered</div>
        </div>
        <div class="metric">
            <div class="metric-value">{queries}</div>
            <div class="metric-label">Queries Executed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{risk_score}</div>
            <div class="metric-label">Risk Score</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Risk Assessment</h2>
        <p>
            Risk Level: 
            <span class="risk-badge risk-{risk_level_lower}">{risk_level}</span>
        </p>
    </div>
    
    <div class="section">
        <h2>Full Report</h2>
        <pre style="white-space: pre-wrap; font-family: inherit;">{report}</pre>
    </div>
</body>
</html>
"""
        
        risk_assessment = results.get("risk_assessment", {})
        html_content = html_template.format(
            target=results["target"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_facts=results["total_facts"],
            queries=results["queries_executed"],
            risk_score=risk_assessment.get("risk_score", 0),
            risk_level=risk_assessment.get("risk_level", "UNKNOWN"),
            risk_level_lower=risk_assessment.get("risk_level", "unknown").lower(),
            report=results["report"]
        )
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)


# ============================================================================
# Example Usage and Integration
# ============================================================================

if __name__ == "__main__":
    print("""
Deep Research AI Agent - Additional Implementation Files
========================================================

This file contains:
1. CLI Interface (cli.py)
2. Test Suite (test_agent.py)
3. Configuration Management (config.py)
4. Utility Functions (utils.py)
5. Export Utilities (export.py)

To use these components:

1. Split this file into separate modules
2. Install requirements: pip install -r requirements.txt
3. Configure .env with API keys
4. Run: python cli.py --target "John Doe"

For testing:
pytest test_agent.py -v

For configuration:
python config.py --help
""")