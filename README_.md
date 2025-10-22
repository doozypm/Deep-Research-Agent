# Deep-Research-Agent
Autonomous research agent capable of conducting comprehensive investigations on individuals or entities to uncover hidden connections, potential risks, and strategic insights

**GETTING STARTED**

# 1. Install dependencies
pip install anthropic openai langgraph httpx networkx pydantic

# 2. Set up environment variables
export ANTHROPIC_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export TAVILY_API_KEY="your_key"
export SERPER_API_KEY="your_key"

# 3. Run the agent
python cli.py --target "John Doe" --output report.md

# 4. Run evaluation
python cli.py --evaluate
