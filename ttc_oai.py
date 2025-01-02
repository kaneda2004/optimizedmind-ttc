import os
import concurrent.futures
import json
import time
import asyncio
from datetime import datetime
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Union, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from openai import OpenAI
from rich.columns import Columns
from rich.console import Group

# Set up logging
log_filename = f"ttc_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# File handler with full debug info
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler with filtered info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Create a filter for the console handler
class ConsoleFilter(logging.Filter):
    def filter(self, record):
        # Filter out raw model responses and detailed logs from console
        filtered_patterns = [
            "Raw model response:",
            "Sending prompt to model:",
            "Parameters:",
            "Extracted content:",
            "Response content:",
            "Verification results for response",
            "Ratings:",
            "Previous response",
            "Using previous responses",
            "Generation parameters:",
            "Creating task",
            "Starting sequential generation",
            "Starting parallel generation",
            "Starting response aggregation",
            "Starting analytical response verification",
            "Starting creative response evaluation",
            "Starting general response evaluation",
            "Number of responses",
            "Selected best",
            "Query:",
            "Starting query processing"
        ]
        return not any(pattern in record.getMessage() for pattern in filtered_patterns)

console_handler.addFilter(ConsoleFilter())

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Token tracking for reasoning and final answer
reasoning_tokens = 0
final_answer_tokens = 0

class QueryType(Enum):
    """Classify different types of queries for optimal processing."""
    ANALYTICAL = "analytical"       # Math, logic problems, analysis
    CREATIVE = "creative"          # Writing, storytelling, art
    INFORMATIONAL = "informational" # Facts, explanations, summaries
    OPEN_ENDED = "open_ended"      # Discussion, opinions, exploration
    PROCEDURAL = "procedural"      # Step-by-step tasks, instructions

class QueryDifficulty(Enum):
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5

@dataclass
class ProcessingStrategy:
    sequential_samples: int
    parallel_samples: int
    temperature: float
    max_tokens: int
    top_p: float
    type_specific_params: Dict
    max_revision_steps: int = 3

class ResponseCache:
    """Cache for storing and retrieving responses."""
    def __init__(self, cache_file: str = "response_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load cache from {self.cache_file}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get(self, key: str) -> Optional[Dict]:
        """Get cached response if it exists."""
        return self.cache.get(key)

    def set(self, key: str, value: Dict):
        """Cache a response."""
        self.cache[key] = value
        self._save_cache()

class EnhancedOpenAIProcessor:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = ResponseCache()
        self.total_tokens = 0  # Add instance variable for token tracking
        
        # Token limits for different difficulty levels
        self.token_limits = {
            QueryDifficulty.VERY_EASY: 4096,
            QueryDifficulty.EASY: 6144,
            QueryDifficulty.MEDIUM: 8192,
            QueryDifficulty.HARD: 12288,
            QueryDifficulty.VERY_HARD: 16383
        }
        
        # Base generation config
        self.generation_config = {
            "temperature": 0.7,
            "max_tokens": 8192,  # This will be dynamically adjusted
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "response_format": {"type": "json_object"}
        }
        
        self.initial_budget = 5
        self.max_budget = 12
        self.current_best_score = 0
        self.budget_increase_threshold = 0.7  # Score threshold for considering a good solution

    def _calculate_dynamic_budget(self, difficulty: QueryDifficulty, current_score: float = 0) -> int:
        """Calculate dynamic budget based on difficulty and current best solution score."""
        # Base budget multipliers for different difficulty levels
        difficulty_multipliers = {
            QueryDifficulty.VERY_EASY: 0.6,
            QueryDifficulty.EASY: 0.8,
            QueryDifficulty.MEDIUM: 1.0,
            QueryDifficulty.HARD: 1.5,
            QueryDifficulty.VERY_HARD: 2.0
        }
        
        # Calculate base budget from difficulty
        base_budget = int(self.initial_budget * difficulty_multipliers[difficulty])
        
        # If we have a current score, adjust budget based on solution quality
        if current_score > 0:
            # If the score is below our threshold, increase budget
            if current_score < self.budget_increase_threshold:
                quality_multiplier = 1 + (self.budget_increase_threshold - current_score)
                base_budget = int(base_budget * quality_multiplier)
        
        # Ensure budget stays within bounds
        return min(max(base_budget, self.initial_budget), self.max_budget)

    async def _classify_query_type(self, query: str) -> QueryType:
        """Determine the type of query for optimal processing."""
        prompt = f"""Classify this query into one of these categories:
        - ANALYTICAL (math, logic, analysis)
        - CREATIVE (writing, storytelling, art)
        - INFORMATIONAL (facts, explanations)
        - OPEN_ENDED (discussion, opinions)
        - PROCEDURAL (step-by-step tasks)
        
        Query: {query}
        
        Response format: Just the category name in lowercase."""
        
        try:
            response = await self._get_completion(prompt, temperature=0.1)
            # Extract the content and convert to lowercase
            category = response['content'].strip().lower()
            
            # Map common variations to valid enum values
            category_mapping = {
                'analytical': 'analytical',
                'creative': 'creative',
                'informational': 'informational',
                'open ended': 'open_ended',
                'open-ended': 'open_ended',
                'procedural': 'procedural'
            }
            
            # Get the standardized category name
            standardized_category = category_mapping.get(category, 'open_ended')
            return QueryType(standardized_category)
            
        except (ValueError, Exception) as e:
            logger.warning(f"Query classification failed: {e}")
            return QueryType.OPEN_ENDED

    async def _estimate_difficulty(self, query: str, query_type: QueryType, num_samples: int = 8) -> QueryDifficulty:
        """Estimate query difficulty with caching and parallel sampling."""
        cache_key = f"difficulty_{hash(query)}"
        cached = self.cache.get(cache_key)
        if cached:
            return QueryDifficulty(cached)

        type_specific_prompts = {
            QueryType.ANALYTICAL: "Rate the computational complexity required...",
            QueryType.CREATIVE: "Rate the creative complexity required...",
            QueryType.INFORMATIONAL: "Rate the research depth required...",
            QueryType.OPEN_ENDED: "Rate the cognitive complexity required...",
            QueryType.PROCEDURAL: "Rate the procedural complexity required..."
        }
        
        base_prompt = type_specific_prompts.get(query_type, "Rate the difficulty...")
        prompt = f"{base_prompt} of this query from 1-5 (1=very easy, 5=very hard):\n\n{query}\n\nResponse format: Just the number (1-5)."

        try:
            responses = []
            for _ in range(num_samples):
                response = await self._get_completion(prompt, temperature=0.7)
                try:
                    # Extract just the number from the response
                    difficulty_str = response['content'].strip()
                    # Remove any markdown formatting or extra text
                    difficulty_str = re.sub(r'[^\d]', '', difficulty_str)
                    difficulty = int(difficulty_str)
                    if 1 <= difficulty <= 5:
                        responses.append(difficulty)
                except ValueError:
                    continue

            if responses:
                median_difficulty = sorted(responses)[len(responses)//2]
                self.cache.set(cache_key, median_difficulty)
                return QueryDifficulty(median_difficulty)

            return QueryDifficulty.MEDIUM

        except Exception as e:
            logger.error(f"Error estimating difficulty: {e}")
            return QueryDifficulty.MEDIUM

    def _get_type_specific_strategy(self, query_type: QueryType, difficulty: QueryDifficulty, total_budget: int) -> ProcessingStrategy:
        """Get optimal processing strategy based on query type and difficulty."""
        type_strategies = {
            QueryType.ANALYTICAL.value: {
                'temperature': 0.3,
                'top_p': 0.95,
                'parallel_weight': 0.7,
            },
            QueryType.CREATIVE.value: {
                'temperature': 0.9,
                'top_p': 0.98,
                'parallel_weight': 0.3,
            },
            QueryType.INFORMATIONAL.value: {
                'temperature': 0.4,
                'top_p': 0.9,
                'parallel_weight': 0.5,
            },
            QueryType.OPEN_ENDED.value: {
                'temperature': 0.8,
                'top_p': 0.95,
                'parallel_weight': 0.4,
            },
            QueryType.PROCEDURAL.value: {
                'temperature': 0.5,
                'top_p': 0.9,
                'parallel_weight': 0.6,
            }
        }

        strategy = type_strategies.get(query_type.value, type_strategies[QueryType.OPEN_ENDED.value])
        
        # Get max tokens based on difficulty
        difficulty_token_limits = {
            QueryDifficulty.VERY_EASY.value: 4096,
            QueryDifficulty.EASY.value: 6144,
            QueryDifficulty.MEDIUM.value: 8192,
            QueryDifficulty.HARD.value: 12288,
            QueryDifficulty.VERY_HARD.value: 16383
        }
        max_tokens = difficulty_token_limits[difficulty.value]
        
        # For harder problems, we want to reserve more tokens for the prompt
        # and leave fewer for completion to avoid hitting the total limit
        prompt_reservation = {
            QueryDifficulty.VERY_EASY.value: 0.2,   # 20% for prompt, 80% for completion
            QueryDifficulty.EASY.value: 0.25,       # 25% for prompt, 75% for completion
            QueryDifficulty.MEDIUM.value: 0.3,      # 30% for prompt, 70% for completion
            QueryDifficulty.HARD.value: 0.35,       # 35% for prompt, 65% for completion
            QueryDifficulty.VERY_HARD.value: 0.4    # 40% for prompt, 60% for completion
        }
        
        # Calculate completion tokens, leaving room for prompt
        completion_tokens = int(max_tokens * (1 - prompt_reservation[difficulty.value]))
        
        # Adjust samples based on difficulty and total budget
        parallel_samples = int(total_budget * strategy['parallel_weight'])
        sequential_samples = total_budget - parallel_samples
        
        # Adjust further if question is harder
        if difficulty in [QueryDifficulty.HARD, QueryDifficulty.VERY_HARD]:
            parallel_samples = int(parallel_samples * 1.2)
            sequential_samples = max(1, total_budget - parallel_samples)
        
        return ProcessingStrategy(
            sequential_samples=sequential_samples,
            parallel_samples=parallel_samples,
            temperature=strategy['temperature'],
            max_tokens=completion_tokens,
            top_p=strategy['top_p'],
            type_specific_params=strategy
        )

    async def _get_completion(self, 
                              prompt: str, 
                              temperature: float = 0.7, 
                              max_tokens: int = None,
                              top_p: float = None,
                              response_format: Dict = None) -> Dict:
        """Get completion from OpenAI API with error handling and retries."""
        max_retries = 3
        retry_delay = 1

        # Log the prompt being sent to file only
        logger.debug("Sending prompt to model:")
        logger.debug(prompt)
        logger.debug(f"Parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, response_format={response_format}")

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens or self.generation_config["max_tokens"],
                    top_p=top_p or self.generation_config["top_p"],
                    response_format=response_format
                )
                
                # Log the full response to file only
                logger.debug("Raw model response:")
                logger.debug(json.dumps(response.model_dump(), indent=2))
                
                # Extract content and token usage
                content = response.choices[0].message.content.strip()
                token_usage = response.usage.total_tokens
                
                logger.debug("Extracted content:")
                logger.debug(content)
                logger.debug(f"Token usage: {token_usage}")
                
                return {
                    'content': content,
                    'token_usage': token_usage
                }

            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise

    async def _generate_sequential(self, 
                                   query: str, 
                                   prev_responses: List[Dict], 
                                   strategy: ProcessingStrategy) -> Dict:
        """Generate improved response based on previous attempts."""
        start_time = time.time()
        
        if prev_responses:
            context = "\n\n".join([
                f"Previous attempt {i+1}:\n{resp['text']}"
                for i, resp in enumerate(prev_responses[-3:])
            ])
            
            prompt = f"""Review these previous attempts and provide an improved response:

Previous attempts:
{context}

Original query:
{query}"""
        else:
            prompt = query

        response = await self._get_completion(
            prompt,
            temperature=strategy.temperature,
            max_tokens=strategy.max_tokens,
            top_p=strategy.top_p
        )
        
        duration = time.time() - start_time
        return {
            'text': response['content'],
            'type': 'Sequential',
            'duration': duration,
            'tokens': response['token_usage']
        }

    async def _generate_parallel(self, 
                                 query: str, 
                                 strategy: ProcessingStrategy) -> List[Dict]:
        """Generate multiple responses in parallel."""
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for i in range(strategy.parallel_samples):
                task = tg.create_task(self._generate_parallel_response(
                    query,
                    strategy,
                    i + 1
                ))
                tasks.append(task)

        results = [t.result() for t in tasks if t.result() is not None]
        return results
        
    async def _generate_parallel_response(self,
                                        query: str,
                                        strategy: ProcessingStrategy,
                                        index: int) -> Dict:
        """Generate a single parallel response with timing."""
        start_time = time.time()
        
        response = await self._get_completion(
            query,
            temperature=strategy.temperature,
            max_tokens=strategy.max_tokens,
            top_p=strategy.top_p
        )
        
        duration = time.time() - start_time
        return {
            'text': response['content'],
            'type': 'Parallel',
            'duration': duration,
            'tokens': response['token_usage']
        }

    async def _aggregate_responses(self, 
                                   responses: List[Dict], 
                                   query_type: QueryType) -> Dict:
        """Aggregate and select best response using type-specific criteria."""
        logger.info(f"Starting response aggregation for query type: {query_type.value}")
        logger.info(f"Number of responses to aggregate: {len(responses)}")
        
        if not responses:
            logger.warning("No responses to aggregate")
            return None

        # Different aggregation strategies for different query types
        if query_type == QueryType.ANALYTICAL:
            logger.info("Using analytical verification strategy")
            return await self._verify_analytical_responses(responses)
        elif query_type == QueryType.CREATIVE:
            logger.info("Using creative evaluation strategy")
            return await self._evaluate_creative_responses(responses)
        else:
            logger.info("Using general evaluation strategy")
            return await self._evaluate_general_responses(responses)

    async def _verify_analytical_responses(self, responses: List[Dict]) -> Dict:
        """Verify analytical responses for correctness and reasoning."""
        logger.info("Starting analytical response verification")
        logger.info(f"Number of responses to verify: {len(responses)}")
        verification_results = []
        
        for i, response_data in enumerate(responses, 1):
            logger.info(f"Verifying response {i}/{len(responses)}")
            logger.info(f"Response content:\n{response_data['text']}")
            
            # Verify each response
            verification_prompt = f"""Evaluate this response and provide a JSON object with the following ratings (0-10):
1. Mathematical/logical correctness
2. Clear reasoning
3. Efficient approach

Response to evaluate:
{response_data['text']}

Return a JSON object with this exact format:
{{"correctness": X, "reasoning": Y, "efficiency": Z}}"""

            try:
                result = await self._get_completion(
                    verification_prompt, 
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                ratings = json.loads(result['content'])
                total_score = (
                    ratings["correctness"] 
                    + ratings["reasoning"] 
                    + ratings["efficiency"]
                ) / 3
                logger.info(f"Total score: {total_score}")
                verification_results.append((response_data['text'], total_score))
            except Exception as e:
                logger.error(f"Verification failed for response {i}: {e}")
                verification_results.append((response_data['text'], 0))

        # Select best response
        best_response = max(verification_results, key=lambda x: x[1])
        logger.info(f"Score: {best_response[1]}")
        
        # Get the final response with token tracking as final answer
        final_response = await self._get_completion(
            f"Given this analytical response, format it clearly and concisely:\n\n{best_response[0]}",
            temperature=0.3
        )
        
        return {
            "response": final_response['content'],
            "score": best_response[1],
            "type": "analytical",
            "tokens": final_response['token_usage']
        }

    async def _evaluate_creative_responses(self, responses: List[Dict]) -> Dict:
        """Evaluate creative responses for originality, coherence, and engagement."""
        logger.info("Starting creative response evaluation")
        logger.info(f"Number of responses to evaluate: {len(responses)}")
        evaluation_results = []
        
        for i, response_data in enumerate(responses, 1):
            logger.info(f"Evaluating response {i}/{len(responses)}")
            logger.info(f"Response content:\n{response_data['text']}")
            
            evaluation_prompt = f"""Evaluate this creative response and provide a JSON object with the following ratings (0-10):
1. Originality
2. Coherence
3. Engagement

Response to evaluate:
{response_data['text']}

Return a JSON object with this exact format:
{{"originality": X, "coherence": Y, "engagement": Z}}"""

            try:
                result = await self._get_completion(
                    evaluation_prompt, 
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                ratings = json.loads(result['content'])
                total_score = (
                    ratings["originality"] 
                    + ratings["coherence"] 
                    + ratings["engagement"]
                ) / 3
                logger.info(f"Total score: {total_score}")
                evaluation_results.append((response_data['text'], total_score))
            except Exception as e:
                logger.error(f"Creative evaluation failed for response {i}: {e}")
                evaluation_results.append((response_data['text'], 0))

        best_response = max(evaluation_results, key=lambda x: x[1])
        logger.info(f"Score: {best_response[1]}")
        
        # Get the final response with token tracking as final answer
        final_response = await self._get_completion(
            f"Given this creative response, format it clearly and engagingly:\n\n{best_response[0]}",
            temperature=0.3
        )
        
        return {
            "response": final_response['content'],
            "score": best_response[1],
            "type": "creative",
            "tokens": final_response['token_usage']
        }

    async def _evaluate_general_responses(self, responses: List[Dict]) -> Dict:
        """Evaluate general responses for relevance, clarity, and completeness."""
        logger.info("Starting general response evaluation")
        logger.info(f"Number of responses to evaluate: {len(responses)}")
        evaluation_results = []
        
        for i, response_data in enumerate(responses, 1):
            logger.info(f"Evaluating response {i}/{len(responses)}")
            logger.info(f"Response content:\n{response_data['text']}")
            
            evaluation_prompt = f"""Evaluate this response and provide a JSON object with the following ratings (0-10):
1. Relevance
2. Clarity
3. Completeness

Response to evaluate:
{response_data['text']}

Return a JSON object with this exact format:
{{"relevance": X, "clarity": Y, "completeness": Z}}"""

            try:
                result = await self._get_completion(
                    evaluation_prompt, 
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                ratings = json.loads(result['content'])
                total_score = (
                    ratings["relevance"] 
                    + ratings["clarity"] 
                    + ratings["completeness"]
                ) / 3
                logger.info(f"Total score: {total_score}")
                evaluation_results.append((response_data['text'], total_score))
            except Exception as e:
                logger.error(f"General evaluation failed for response {i}: {e}")
                evaluation_results.append((response_data['text'], 0))

        best_response = max(evaluation_results, key=lambda x: x[1])
        logger.info(f"Score: {best_response[1]}")
        
        # Get the final response with token tracking as final answer
        final_response = await self._get_completion(
            f"Given this response, format it clearly and concisely:\n\n{best_response[0]}",
            temperature=0.3
        )
        
        return {
            "response": final_response['content'],
            "score": best_response[1],
            "type": "general",
            "tokens": final_response['token_usage']
        }

    def format_response_table(self, responses: List[Dict], start_time: float) -> Table:
        """Format responses into a Rich table."""
        table = Table(
            title="Generated Responses", 
            show_header=True, 
            header_style="bold magenta"
        )
        table.add_column("№", style="dim")
        table.add_column("Type")
        table.add_column("Response Preview", width=60)
        table.add_column("Time", style="dim")
        
        for i, response_data in enumerate(responses, 1):
            # Get a preview of the response, truncated if too long
            response_text = response_data['text']
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            
            table.add_row(
                str(i),
                response_data['type'],
                Text(preview, no_wrap=False),
                f"{response_data['duration']:.2f}s"
            )
        
        return table

    def format_result_panel(self, result: Dict, duration: float) -> Panel:
        """Format the final result into a Rich panel with side-by-side layout."""
        if "error" in result:
            content = [
                Text("❌ Error", style="bold red"),
                result["error"]
            ]
            return Panel("\n".join(str(line) for line in content), title="Error", border_style="red")

        # Calculate metrics
        budget_percent = (result['total_budget_used'] / self.max_budget) * 100
        difficulty_token_limits = {
            QueryDifficulty.VERY_EASY.value: 4096,
            QueryDifficulty.EASY.value: 6144,
            QueryDifficulty.MEDIUM.value: 8192,
            QueryDifficulty.HARD.value: 12288,
            QueryDifficulty.VERY_HARD.value: 16383
        }
        
        difficulty_name = result['difficulty'].upper()
        try:
            difficulty_enum = QueryDifficulty[difficulty_name]
            token_limit = difficulty_token_limits[difficulty_enum.value]
        except KeyError:
            logger.error(f"Unknown difficulty level: {difficulty_name}")
            token_limit = 8192

        total_calls = len(result.get('all_responses', [])) + 1
        avg_tokens_per_call = result.get('total_tokens', 0) / total_calls if total_calls > 0 else 0
        token_percent = (avg_tokens_per_call / token_limit) * 100 if token_limit > 0 else 0

        # Create Analysis panel
        analysis_content = Group(
            Text("📊 Analysis", style="bold blue"),
            Text(f"Query Type: {result['query_type']}"),
            Text(f"Difficulty: {result['difficulty']}"),
            Text(f"Strategy: Sequential={result['strategy']['sequential_samples']}, Parallel={result['strategy']['parallel_samples']}"),
            Text(f"Budget Used: {result['total_budget_used']}/{self.max_budget} ({budget_percent:.1f}%)"),
            Text(f"Total Duration: {duration:.2f}s")
        )
        analysis_panel = Panel(analysis_content, border_style="blue", width=60)

        # Create Token Usage panel
        token_content = Group(
            Text("🔢 Token Usage", style="bold cyan"),
            Text(f"Token Limit per Call: {token_limit:,} tokens"),
            Text(f"Average Tokens per Call: {avg_tokens_per_call:,.0f} tokens ({token_percent:.1f}%)"),
            Text(f"Total API Calls: {total_calls}"),
            Text(f"Total Tokens Used: {result.get('total_tokens', 0):,} tokens"),
            Text("")  # Add padding to match analysis panel height
        )
        token_panel = Panel(token_content, border_style="cyan", width=60)

        # Create Best Response panel
        response_content = Group(
            Text("🎯 Best Response", style="bold green"),
            Text(result['best_response']['response'], style="yellow"),
            Text(f"\nScore: {result['best_response']['score']:.2f}"),
            Text(f"Type: {result['best_response']['type']}")
        )
        response_panel = Panel(response_content, border_style="green", width=120)

        # Combine all panels
        return Panel(
            Group(
                Columns([analysis_panel, token_panel]),
                response_panel
            ),
            title="Final Results",
            border_style="bright_blue"
        )

    async def process_query(self, query: str, total_budget: int = None) -> Dict:
        """Main processing pipeline for any type of query."""
        start_time = time.time()
        total_tokens = 0
        
        try:
            # 1. Classify query type
            query_type_raw = await self._get_completion(
                f"""Classify this query into one of these categories:
                - ANALYTICAL (math, logic, analysis)
                - CREATIVE (writing, storytelling, art)
                - INFORMATIONAL (facts, explanations)
                - OPEN_ENDED (discussion, opinions)
                - PROCEDURAL (step-by-step tasks)
                
                Query: {query}
                
                Response format: Just the category name in lowercase.""",
                temperature=0.1
            )
            total_tokens += query_type_raw['token_usage']
            self.total_tokens = total_tokens  # Update the instance variable
            
            # Map the raw response to a QueryType enum
            category = query_type_raw['content'].strip().lower()
            category_mapping = {
                'analytical': 'analytical',
                'creative': 'creative',
                'informational': 'informational',
                'open ended': 'open_ended',
                'open-ended': 'open_ended',
                'procedural': 'procedural'
            }
            standardized_category = category_mapping.get(category, 'open_ended')
            query_type = QueryType(standardized_category)
            
            # 2. Estimate difficulty
            difficulty_raw = await self._get_completion(
                f"""Rate the computational complexity required of this query from 1-5 (1=very easy, 5=very hard):

                Query: {query}
                
                Response format: Just the number (1-5).""",
                temperature=0.7
            )
            total_tokens += difficulty_raw['token_usage']
            self.total_tokens = total_tokens  # Update the instance variable
            
            # Extract just the number from the response
            difficulty_str = difficulty_raw['content'].strip()
            # Remove any markdown formatting or extra text
            difficulty_str = re.sub(r'[^\d]', '', difficulty_str)
            difficulty = QueryDifficulty(int(difficulty_str))
            
            # Calculate initial budget if not provided
            if total_budget is None:
                total_budget = self._calculate_dynamic_budget(difficulty)
            
            # 3. Get processing strategy
            strategy = self._get_type_specific_strategy(query_type, difficulty, total_budget)
            
            # 4. Generate responses
            all_responses = []
            
            # Sequential generation
            if strategy.sequential_samples > 0:
                sequential_responses = []
                for i in range(strategy.sequential_samples):
                    response = await self._generate_sequential(
                        query,
                        sequential_responses,
                        strategy
                    )
                    sequential_responses.append(response)
                    all_responses.append(response)
                    total_tokens += response['tokens']
                    self.total_tokens = total_tokens  # Update the instance variable
            
            # Parallel generation
            if strategy.parallel_samples > 0:
                parallel_responses = await self._generate_parallel(query, strategy)
                for response in parallel_responses:
                    all_responses.append(response)
                    total_tokens += response['tokens']
                    self.total_tokens = total_tokens  # Update the instance variable
            
            # 5. Aggregate and select best response
            best_response = await self._aggregate_responses(all_responses, query_type)
            if best_response:
                total_tokens += best_response.get('tokens', 0)
                self.total_tokens = total_tokens  # Update the instance variable
            
            duration = time.time() - start_time
            
            result = {
                "query_type": query_type.value,
                "difficulty": difficulty.name,
                "strategy": {
                    "sequential_samples": strategy.sequential_samples,
                    "parallel_samples": strategy.parallel_samples,
                    "temperature": strategy.temperature,
                    "top_p": strategy.top_p
                },
                "best_response": best_response,
                "all_responses": all_responses,
                "processing_time": duration,
                "total_budget_used": total_budget,
                "total_tokens": total_tokens
            }
            return result
            
        except Exception as e:
            error_msg = f"Error in process_query: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "processing_time": time.time() - start_time,
                "total_tokens": total_tokens
            }

async def main():
    console = Console()
    logger.info("Starting new run")
    
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        processor = EnhancedOpenAIProcessor(api_key)
        
        # Example queries of different types
        default_queries = [
            ("Analytical", "Find the volume of a cone with radius 5 and height 12."),
            ("Creative", "Write a short story about a cat discovering a magical portal."),
            ("Informational", "Explain how photosynthesis works in simple terms."),
            ("Open-ended", "What are your thoughts on the future of artificial intelligence?"),
            ("Procedural", "How do I make a perfect omelet?"),
            ("Complex Math", "Given a sphere inscribed in a cylinder, which is inscribed in a cone with height h and base radius r, "
             "find the ratio of the volumes of the cone to cylinder to sphere in terms of h and r, and prove that "
             "this ratio is constant regardless of the dimensions. Then find the minimum value of h/r that makes "
             "this configuration possible.")
        ]

        # Create a pretty selection menu
        console.print(r"""
[bold cyan]                           $$\      $$\     
                           $$$\    $$$ |    
                  $$$$$$\  $$$$\  $$$$ |    
                 $$  __$$\ $$\$$\$$ $$ |    
                 $$ /  $$ |$$ \$$$  $$ |    
                 $$ |  $$ |$$ |\$  /$$ |    
              $$\\$$$$$$  |$$ | \_/ $$ |$$\ 
              \__|\_____/ \__|     \__|\__|


             _   _           _             ____  ____           _ 
            | | (_)         (_)           | |  \/  (_)         | |
  ___  _ __ | |_ _ _ __ ___  _ _______  __| | .  . |_ _ __   __| |
 / _ \| '_ \| __| | '_ ` _ \| |_  / _ \/ _` | |\/| | | '_ \ / _` |
| (_) | |_) | |_| | | | | | | |/ |  __| (_| | |  | | | | | | (_| |
 \___/| .__/ \__|_|_| |_| |_|_/___\___|\__,_\_|  |_|_|_| |_|\__,_|
      | |                                                         
      |_|                                                         [/bold cyan]
""")
        console.print("[bold cyan]DIYing Test Time Compute since 2024[/bold cyan]")
        console.print("\n[yellow]Please select a query type or enter your own:[/yellow]")
        console.print("\n[bold green]Default Queries:[/bold green]")
        
        for i, (category, query) in enumerate(default_queries, 1):
            console.print(f"[cyan]{i}.[/cyan] [bold]{category}[/bold]")
            console.print(f"   {query[:100]}..." if len(query) > 100 else f"   {query}")
            console.print()
            
        console.print("[cyan]C.[/cyan] [bold]Custom Query[/bold]\n")
        
        while True:
            choice = console.input("[yellow]Enter your choice (1-6 or C): [/yellow]").strip().upper()
            
            if choice == 'C':
                query = console.input("\n[yellow]Enter your query: [/yellow]").strip()
                if query:
                    break
            elif choice.isdigit() and 1 <= int(choice) <= len(default_queries):
                query = default_queries[int(choice)-1][1]
                break
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
        
        console.print("\n[bold green]Processing your query...[/bold green]\n")
        
        start_time = time.time()
        with console.status("[bold blue]") as status:
            async def update_status():
                while True:
                    elapsed = time.time() - start_time
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    status.update(f"[bold blue]Analyzing query for {minutes:02d}m {seconds:02d}s... ({processor.total_tokens:,} tokens used)")
                    await asyncio.sleep(0.1)
                
            # Start the status updater task
            status_task = asyncio.create_task(update_status())
            
            # Process the query
            console.print(Panel(query, title="Question", border_style="yellow"))
            result = await processor.process_query(query)
            duration = result.get("processing_time", 0)
            
            # Cancel the status updater
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass
            
            console.print("\n")
            responses_table = processor.format_response_table(
                result.get("all_responses", []), 
                time.time() - duration
            )
            console.print(responses_table)
            
            console.print("\n")
            result_panel = processor.format_result_panel(result, duration)
            console.print(result_panel)
            
    except Exception as e:
        error_msg = f"Error in main execution: {str(e)}"
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        logger.error(error_msg)
    finally:
        if 'processor' in locals():
            del processor
        logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())