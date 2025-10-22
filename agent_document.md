# Deep Research AI Agent - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Evaluation Framework](#evaluation-framework)
7. [Advanced Features](#advanced-features)
8. [Prompt Engineering](#prompt-engineering)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Deep Research AI Agent is an autonomous intelligence gathering system designed for comprehensive entity investigations. It combines multiple AI models, strategic search progression, and sophisticated analysis techniques to uncover hidden connections, assess risks, and generate actionable intelligence reports.

### Key Features

- **Multi-Model Integration**: Leverages Claude Sonnet 4.5, Claude Opus 4, and GPT-4.1 for diverse perspectives
- **Intelligent Search Progression**: 5-phase strategy that builds on previous discoveries
- **Dynamic Query Refinement**: Adapts search strategy based on findings
- **Fact Verification**: Cross-references information across multiple sources
- **Risk Pattern Recognition**: Identifies red flags and concerning associations
- **Connection Mapping**: Builds network graphs of relationships
- **Comprehensive Reporting**: Generates detailed assessment reports with confidence scoring

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Deep Research Agent                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Query          │  │ Search         │  │ Fact         │ │
│  │ Orchestrator   │→ │ Engine         │→ │ Extractor    │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│          ↓                                       ↓          │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Connection     │← │ Risk           │← │ Model        │ │
│  │ Mapper         │  │ Analyzer       │  │ Manager      │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│          ↓                                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │            Report Generator                         │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### LangGraph Workflow

```
Initialize → Generate Queries → Execute Search → Extract Facts
                    ↑                                    ↓
                    |                              Analyze Risks
                    |                                    ↓
            (Continue/Finish) ← Map Connections ← (Decision Point)
                    ↓
            Generate Report → END
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- API keys for:
  - Anthropic (Claude)
  - OpenAI (GPT)
  - Tavily (search)
  - Serper (search)

### Setup

```bash
# Clone or create project directory
mkdir deep-research-agent
cd deep-research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install anthropic openai httpx langgraph networkx pydantic

# Install optional dependencies
pip install spacy python-dotenv redis
python -m spacy download en_core_web_sm
```

### Environment Configuration

Create a `.env` file:

```env
# AI Model APIs
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Search APIs
TAVILY_API_KEY=your_tavily_key_here
SERPER_API_KEY=your_serper_key_here

# Optional: Rate Limiting
REDIS_URL=redis://localhost:6379

# Optional: Database
DATABASE_URL=postgresql://user:pass@localhost/research_db
```

---

## Configuration

### Model Configuration

Edit `config.py`:

```python
MODEL_CONFIG = {
    "primary_model": "claude-sonnet-4-5-20250929",
    "deep_analysis_model": "claude-opus-4-20250514",
    "alternative_model": "gpt-4-turbo-preview",
    "temperature": 0.3,  # Lower for factual accuracy
    "max_tokens": 4096
}

SEARCH_CONFIG = {
    "max_results_per_query": 15,
    "depth_levels": {
        "surface": "basic",
        "medium": "basic",
        "deep": "advanced",
        "maximum": "advanced"
    },
    "cache_ttl": 3600  # 1 hour
}

AGENT_CONFIG = {
    "max_phases": 5,
    "min_facts_per_phase": 20,
    "max_queries_per_phase": 5,
    "verification_threshold": 0.7,
    "rate_limit_per_minute": 50
}
```

---

## Usage

### Basic Usage

```python
import asyncio
from deep_research_agent import DeepResearchAgent

async def main():
    agent = DeepResearchAgent()
    
    # Research a single entity
    results = await agent.research("John Doe")
    
    # Access results
    print(f"Report: {results['report']}")
    print(f"Facts discovered: {results['total_facts']}")
    print(f"Risk level: {results['risk_assessment']['risk_level']}")
    
    # Save report
    with open("report_john_doe.md", "w") as f:
        f.write(results["report"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

```python
# Custom search depth
agent = DeepResearchAgent()
agent.config.max_phases = 7  # Deeper investigation

# Research with specific focus
results = await agent.research(
    target_name="Jane Smith",
    focus_areas=["financial", "legal", "connections"],
    depth_override=SearchDepth.MAXIMUM
)

# Batch processing
targets = ["Person A", "Person B", "Person C"]
results = await asyncio.gather(*[
    agent.research(target) for target in targets
])
```

### CLI Usage

```bash
# Single target
python -m deep_research_agent --target "John Doe" --output report.md

# With options
python -m deep_research_agent \
  --target "Jane Smith" \
  --phases 7 \
  --depth maximum \
  --focus financial,legal \
  --output jane_smith_report.md

# Batch mode
python -m deep_research_agent --batch targets.txt --output-dir ./reports/

# Evaluation mode
python -m deep_research_agent --evaluate --personas test_personas.json
```

---

## Evaluation Framework

### Test Personas

The evaluation framework uses carefully designed test personas with hidden facts of varying difficulty:

```python
TEST_PERSONAS = [
    {
        "name": "Sarah Chen",
        "type": "Tech Executive",
        "hidden_facts": [
            {
                "fact": "Changed name from Sarah Zhang in 2010",
                "difficulty": "hard",
                "search_hints": ["name change", "legal records", "historical data"]
            },
            {
                "fact": "Silent partner in failed CloudStart startup 2015",
                "difficulty": "very_hard",
                "search_hints": ["business registrations", "SEC filings", "startup databases"]
            },
            {
                "fact": "Family connection to SEC official",
                "difficulty": "extreme",
                "search_hints": ["family tree", "social connections", "biographical data"]
            }
        ],
        "expected_risk": "MEDIUM"
    }
]
```

### Evaluation Metrics

1. **Discovery Rate**: Percentage of hidden facts found
2. **Accuracy**: Correctness of discovered information
3. **Depth Score**: How many "extreme" difficulty facts were found
4. **Efficiency**: Facts discovered per query executed
5. **Risk Assessment Accuracy**: Correct risk level identification
6. **Connection Completeness**: Percentage of expected connections mapped

### Running Evaluations

```python
from deep_research_agent import EvaluationFramework

evaluator = EvaluationFramework()

# Run full evaluation
results = await evaluator.run_full_evaluation()

# Analyze results
print(f"Overall Score: {results['overall_score']:.2%}")
print(f"Average Discovery Rate: {results['avg_discovery_rate']:.2%}")
print(f"Risk Assessment Accuracy: {results['risk_accuracy']:.2%}")

# Detailed breakdown
for persona_result in results['persona_results']:
    print(f"\n{persona_result['name']}:")
    print(f"  Hidden facts found: {persona_result['found']}/{persona_result['total']}")
    print(f"  Queries needed: {persona_result['queries']}")
    print(f"  Efficiency: {persona_result['efficiency']:.2f} facts/query")
```

### Performance Benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| Discovery Rate (Hard) | >70% | TBD |
| Discovery Rate (Very Hard) | >50% | TBD |
| Discovery Rate (Extreme) | >30% | TBD |
| Risk Assessment Accuracy | >80% | TBD |
| Connection Completeness | >75% | TBD |
| Avg Queries per Investigation | <50 | TBD |

---

## Advanced Features

### 1. Custom Search Strategies

```python
class CustomSearchStrategy:
    """Define custom search progression"""
    
    def __init__(self):
        self.phases = [
            {
                "name": "Deep Dive Financial",
                "focus": ["SEC filings", "corporate records", "tax documents"],
                "depth": SearchDepth.MAXIMUM,
                "queries_per_area": 3
            },
            {
                "name": "Network Analysis",
                "focus": ["LinkedIn connections", "board memberships", "partnerships"],
                "depth": SearchDepth.DEEP,
                "queries_per_area": 4
            }
        ]

# Use custom strategy
agent = DeepResearchAgent(search_strategy=CustomSearchStrategy())
```

### 2. Source Validation

```python
class SourceValidator:
    """Validates source credibility"""
    
    TRUSTED_DOMAINS = [
        "sec.gov",
        "companieshouse.gov.uk",
        "linkedin.com",
        "bloomberg.com",
        "reuters.com"
    ]
    
    def validate_source(self, url: str) -> float:
        """Return confidence score for source"""
        domain = extract_domain(url)
        
        if domain in self.TRUSTED_DOMAINS:
            return 0.95
        elif is_news_site(domain):
            return 0.75
        elif is_social_media(domain):
            return 0.50
        else:
            return 0.30
```

### 3. Real-time Monitoring

```python
class ResearchMonitor:
    """Monitor research progress in real-time"""
    
    def __init__(self):
        self.callbacks = []
    
    def on_query_executed(self, query: str, results_count: int):
        """Called after each search"""
        print(f"✓ Executed: {query} ({results_count} results)")
    
    def on_fact_discovered(self, fact: Fact):
        """Called when a new fact is extracted"""
        print(f"  → {fact.category}: {fact.content[:50]}...")
    
    def on_phase_complete(self, phase: int, facts_count: int):
        """Called when a phase completes"""
        print(f"\n✓ Phase {phase} complete: {facts_count} facts discovered\n")

# Use monitor
agent = DeepResearchAgent()
agent.add_monitor(ResearchMonitor())
```

### 4. Export Formats

```python
# Export to JSON
results_json = agent.export_json(results)

# Export to structured data
results_structured = agent.export_structured(results)
# Returns: pandas DataFrame with all facts

# Export connection graph
graph = agent.export_graph(results, format="graphml")
# Can be imported into Gephi, Cytoscape, etc.

# Export timeline
timeline = agent.export_timeline(results)
# Returns: chronological view of all events
```

---

## Prompt Engineering

### Core Principles

The agent uses carefully engineered prompts following these principles:

1. **Specificity**: Clear, detailed instructions
2. **Context**: Relevant background information
3. **Structure**: Expected output format (usually JSON)
4. **Examples**: Few-shot learning when beneficial
5. **Constraints**: Explicit limitations and requirements

### Example: Fact Extraction Prompt

```
ROLE: You are an expert fact extractor for intelligence gathering.

TASK: Extract all relevant facts from the provided search results.

CONTEXT:
- Target: {target_name}
- Query: {query}
- Phase: {phase_number}
- Previous discoveries: {summary_of_known_facts}

INPUT:
{search_results}

REQUIREMENTS:
1. Extract only verifiable facts (not speculation)
2. Include source URL for each fact
3. Assign confidence score (0.0-1.0) based on:
   - Source credibility
   - Cross-reference availability
   - Information specificity
4. Categorize each fact:
   - biographical: Personal information
   - professional: Career, positions, companies
   - financial: Money, investments, compensation
   - legal: Lawsuits, violations, investigations
   - connection: Relationships, associations
   - risk: Red flags, concerns, anomalies
5. Flag temporal information (dates, timelines)
6. Identify contradictions with known facts

OUTPUT FORMAT (JSON):
[
  {
    "content": "Specific factual statement",
    "source": "URL",
    "confidence": 0.85,
    "category": "professional",
    "temporal_info": "2015-2020",
    "red_flag": false,
    "reasoning": "Why this confidence score"
  }
]

EXAMPLES:
[Provide 2-3 examples of well-extracted facts]

Extract facts now:
```

### Prompt Testing

```python
class PromptTester:
    """Test and optimize prompts"""
    
    async def test_prompt_variants(self, base_prompt: str, variants: List[str], 
                                   test_data: List[Dict]) -> Dict:
        """Compare prompt performance"""
        results = {}
        
        for variant_name, variant_prompt in variants:
            scores = []
            for test_case in test_data:
                response = await self.model_manager.query_claude(
                    variant_prompt.format(**test_case)
                )
                score = self.evaluate_response(response, test_case['expected'])
                scores.append(score)
            
            results[variant_name] = {
                'avg_score': sum(scores) / len(scores),
                'scores': scores
            }
        
        return results

# Usage
tester = PromptTester()
variants = [
    ("detailed", prompt_v1),
    ("concise", prompt_v2),
    ("structured", prompt_v3)
]
results = await tester.test_prompt_variants(base_prompt, variants, test_cases)
```

---

## Best Practices

### 1. Rate Limiting

```python
# Implement exponential backoff
class RateLimiter:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
    
    async def execute_with_backoff(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except RateLimitError:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
        raise Exception("Max retries exceeded")
```

### 2. Caching Strategy

```python
# Cache search results
class SearchCache:
    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, query: str) -> Optional[List[Dict]]:
        if query in self.cache:
            cached_time, results = self.cache[query]
            if (datetime.now() - cached_time).seconds < self.ttl:
                return results
        return None
    
    def set(self, query: str, results: List[Dict]):
        self.cache[query] = (datetime.now(), results)
```

### 3. Error Handling

```python
class RobustResearchAgent(DeepResearchAgent):
    """Enhanced error handling"""
    
    async def execute_search_node(self, state: AgentState) -> AgentState:
        try:
            return await super().execute_search_node(state)
        except ConnectionError as e:
            logger.error(f"Network error: {e}")
            state["error_log"].append(f"Network: {str(e)}")
            await asyncio.sleep(5)  # Wait and retry
            return await super().execute_search_node(state)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            state["error_log"].append(f"Error: {str(e)}")
            return state  # Continue with partial results
```

### 4. Data Validation

```python
from pydantic import BaseModel, validator

class ValidatedFact(BaseModel):
    content: str
    source: str
    confidence: float
    category: str
    
    @validator('confidence')
    def confidence_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @validator('category')
    def valid_category(cls, v):
        valid = ['biographical', 'professional', 'financial', 'legal', 'connection', 'risk']
        if v not in valid:
            raise ValueError(f'Category must be one of {valid}')
        return v
```

### 5. Logging and Auditing

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure comprehensive logging
def setup_logging():
    logger = logging.getLogger('deep_research_agent')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = RotatingFileHandler(
        'research_agent.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatting
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
```

---

## Troubleshooting

### Common Issues

#### 1. API Rate Limits

**Problem**: `RateLimitError: Too many requests`

**Solutions**:
```python
# Increase wait time between requests
agent.config.request_delay = 2.0  # seconds

# Reduce concurrent requests
agent.config.max_concurrent = 3

# Use Redis for distributed rate limiting
from redis import Redis
agent.rate_limiter = RedisRateLimiter(Redis())
```

#### 2. Low Quality Results

**Problem**: Agent finds too few facts or low confidence scores

**Solutions**:
```python
# Increase search depth
agent.config.default_depth = SearchDepth.MAXIMUM

# More queries per phase
agent.config.queries_per_phase = 7

# Lower confidence threshold
agent.config.min_confidence = 0.5

# Use more diverse search sources
agent.search_engine.add_source(DuckDuckGoSearch())
agent.search_engine.add_source(BingSearch())
```

#### 3. Model Timeouts

**Problem**: `TimeoutError: Model request exceeded time limit`

**Solutions**:
```python
# Increase timeout
agent.model_manager.timeout = 60.0

# Reduce max tokens
agent.model_manager.max_tokens = 2048

# Use faster model for initial passes
agent.config.phase1_model = "claude-sonnet-4-5-20250929"
agent.config.deep_analysis_model = "claude-opus-4-20250514"
```

#### 4. Memory Issues

**Problem**: `MemoryError` or high RAM usage

**Solutions**:
```python
# Clear cache periodically
agent.search_engine.clear_cache()

# Limit fact history
agent.config.max_facts_in_memory = 1000

# Stream results to disk
agent.config.stream_mode = True
agent.config.output_file = "streaming_results.jsonl"
```

### Debug Mode

```python
# Enable comprehensive debugging
agent = DeepResearchAgent(debug=True)

# This will:
# 1. Log all prompts and responses
# 2. Save intermediate results
# 3. Generate timing reports
# 4. Create visualization of search progression

# Access debug data
debug_data = agent.get_debug_info()
print(f"Total API calls: {debug_data['api_calls']}")
print(f"Average response time: {debug_data['avg_response_time']}")
print(f"Cache hit rate: {debug_data['cache_hit_rate']}")
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile the research process
profiler = cProfile.Profile()
profiler.enable()

results = await agent.research("John Doe")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 time-consuming functions
```

---

## Additional Resources

### Configuration Files

**requirements.txt**
```
anthropic>=0.18.0
openai>=1.12.0
langgraph>=0.0.20
httpx>=0.26.0
networkx>=3.2
pydantic>=2.5.0
python-dotenv>=1.0.0
redis>=5.0.0
spacy>=3.7.0
pandas>=2.1.0
matplotlib>=3.8.0
```

**docker-compose.yml**
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: research_db
      POSTGRES_USER: research_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  agent:
    build: .
    depends_on:
      - redis
      - postgres
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://research_user:secure_password@postgres/research_db
    env_file:
      - .env
    volumes:
      - ./reports:/app/reports
      - ./logs:/app/logs

volumes:
  redis_data:
  postgres_data:
```

### API Documentation

Full API documentation available at: `/docs/api.md`

### Contributing

See `CONTRIBUTING.md` for guidelines on:
- Code style
- Testing requirements
- Pull request process
- Issue reporting

### License

This project is licensed under the MIT License - see `LICENSE` file for details.

### Support

- GitHub Issues: `github.com/doozypm/Deep-Research-Agent/issues`

---

## Appendix

### A. Search Query Templates

```python
QUERY_TEMPLATES = {
    "biographical": [
        "{name} biography background",
        "{name} education history",
        "{name} birth date location"
    ],
    "professional": [
        "{name} employment history",
        "{name} company board member",
        "{name} professional experience"
    ],
    "financial": [
        "{name} SEC filings",
        "{name} financial disclosure",
        "{name} investment portfolio"
    ],
    "legal": [
        "{name} lawsuit litigation",
        "{name} regulatory violation",
        "{name} legal proceedings"
    ]
}
```

### B. Risk Scoring Matrix

| Factor | Weight | Score Calculation |
|--------|--------|-------------------|
| Legal Issues | 0.30 | # of cases × severity |
| Financial Irregularities | 0.25 | # of incidents × impact |
| Reputational Concerns | 0.20 | # of events × visibility |
| Network Risks | 0.15 | # of risky connections × proximity |
| Compliance Issues | 0.10 | # of violations × recency |

### C. Model Comparison

| Model | Speed | Accuracy | Cost | Best For |
|-------|-------|----------|------|----------|
| Claude Sonnet 4.5 | Fast | High | Medium | General research |
| Claude Opus 4 | Slow | Highest | High | Deep analysis |
| GPT-4.1 | Medium | High | Medium | Verification |
| Gemini 2.5 | Fast | Medium | Low | Initial scanning |

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-21  
**Author**: Pratyush Mishra
