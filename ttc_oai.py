import os
import concurrent.futures
import json
import time
import asyncio
import csv
import uuid
from datetime import datetime
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Union, Any, Tuple

import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.console import Group
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

from openai import OpenAI

import io  # For rendering Rich panels to string

# --- Configuration and Setup ---
LOG_FILENAME_PREFIX = "ttc_run"
CONSOLE_LOG_LEVEL = logging.WARNING
FILE_LOG_LEVEL = logging.DEBUG
MODEL_NAME = "gpt-4o-mini"
RESPONSE_CACHE_FILE = "response_cache.json"

INITIAL_BUDGET = 5
MAX_BUDGET = 12
BUDGET_INCREASE_THRESHOLD = 0.7

GENERATION_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 8192,
    "top_p": 0.95,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "response_format": {"type": "json_object"}
}

log_filename = f"{LOG_FILENAME_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(FILE_LOG_LEVEL)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(CONSOLE_LOG_LEVEL)
console_handler.setFormatter(logging.Formatter('%(message)s'))

class ConsoleFilter(logging.Filter):
    def filter(self, record):
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
            "Starting query processing",
            "Verification complete (debug only).",
        ]
        return not any(pattern in record.getMessage() for pattern in filtered_patterns)

console_handler.addFilter(ConsoleFilter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


@dataclass
class ProcessingStrategy:
    sequential_samples: int
    parallel_samples: int
    temperature: float
    max_tokens: int
    top_p: float
    max_revision_steps: int = 3


class ResponseCache:
    def __init__(self, cache_file: str = RESPONSE_CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load cache from {self.cache_file}")
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get(self, key: str) -> Optional[Dict]:
        return self.cache.get(key)

    def set(self, key: str, value: Dict):
        self.cache[key] = value
        self._save_cache()


# ---------------------------------------------------------------------------------------
# Verification Pipeline
# ---------------------------------------------------------------------------------------
class VerificationPipeline:
    def __init__(self, processor):
        self.processor = processor

    async def extract_claims(self, response: str) -> List[str]:
        prompt = f"""Analyze this response and extract all key claims, assertions, or conclusions made.
Return your response as a JSON array with exactly this structure:
[
    "claim1",
    "claim2",
    ...
]

Include both explicit and implicit claims.

Response:
{response}
"""
        result = await self.processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )

        cleaned_content = self.processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse claims extraction response.")
            return []

    async def verify_claim(self, claim: str, context: str) -> Dict:
        prompt = f"""Critically examine this claim in the given context. Consider:
1. Logical validity
2. Mathematical correctness (if applicable) 
3. Supporting evidence
4. Potential counterexamples
5. Hidden assumptions
6. Edge cases

Claim: {claim}
Context: {context}

Return your analysis as a JSON object with exactly this structure:
{{
    "validity": <score 0-10>,
    "correctness": <score 0-10>,
    "completeness": <score 0-10>,
    "issues": ["list", "of", "issues"],
    "suggestions": ["list", "of", "fixes"]
}}
"""
        result = await self.processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )

        cleaned_content = self.processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse claim verification response.")
            return {}

    async def check_relationships(self, response: str) -> Dict:
        prompt = f"""Analyze the logical relationships and dependencies between claims in this response.
Identify any:
1. Contradictions
2. Circular reasoning
3. Missing links in logic
4. Unsubstantiated assumptions

Response:
{response}

Return your analysis as a JSON object with exactly this structure:
{{
    "contradictions": ["contradiction1", "contradiction2", ...],
    "circular_reasoning": ["instance1", "instance2", ...],
    "logic_gaps": ["gap1", "gap2", ...],
    "assumptions": ["assumption1", "assumption2", ...]
}}
"""
        result = await self.processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )

        cleaned_content = self.processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse relationships checking response.")
            return {}

    async def validate_corner_cases(self, response: str) -> Dict:
        prompt = f"""Consider edge cases and special conditions for this response.
Check for:
1. Boundary conditions
2. Degenerate cases
3. Extreme values
4. Special cases

Response:
{response}

Return your analysis as a JSON object with exactly this structure:
{{
    "edge_cases": ["case1", "case2", ...],
    "boundary_conditions": ["condition1", "condition2", ...],
    "special_cases": ["special1", "special2", ...],
    "potential_issues": ["issue1", "issue2", ...]
}}
"""
        result = await self.processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )

        cleaned_content = self.processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse corner cases validation response.")
            return {}

    async def cross_validate(self, responses: List[str]) -> Dict:
        responses_formatted = "\n\n".join(
            [f"Response {i+1}:\n{resp}" for i, resp in enumerate(responses, 1)]
        )
        prompt = f"""Compare these responses and analyze:
1. Agreement/disagreement on key points
2. Different approaches/methodologies used
3. Complementary insights
4. Conflicting conclusions

Responses:
{responses_formatted}

Return your analysis as a JSON object with exactly this structure:
{{
    "agreements": ["agreement1", "agreement2", ...],
    "disagreements": ["disagreement1", "disagreement2", ...],
    "methodology_differences": ["difference1", "difference2", ...],
    "synthesis_opportunities": ["opportunity1", "opportunity2", ...]
}}
"""
        result = await self.processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )

        cleaned_content = self.processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse cross-validation response.")
            return {}

# ---------------------------------------------------------------------------------------
# ResponseAnalyzer
# ---------------------------------------------------------------------------------------
class ResponseAnalyzer:
    @staticmethod
    async def extract_mathematical_steps(processor, text: str) -> List[Dict]:
        prompt = f"""Analyze this response and extract each mathematical step or claim.
For each step:
1. Identify the operation/transformation
2. Check if intermediate results follow
3. Verify the logic connecting steps

Return your analysis as a JSON array with exactly this structure:
[
    {{
        "step": "description of the step",
        "operation": "specific operation performed",
        "input": "input values or expressions",
        "output": "output or result",
        "justification": "explanation of why this step is valid"
    }},
    ...
]

Response:
{text}
"""
        result = await processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        cleaned_content = processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse mathematical steps response.")
            return []

    @staticmethod
    async def check_logical_flow(processor, text: str) -> Dict:
        prompt = f"""Analyze the logical structure and flow of this response.
Check for:
1. Valid premises and conclusions
2. Sound logical connections
3. Complete argument chains
4. Proper support for claims

Return your analysis as a JSON object with exactly this structure:
{{
    "premises": ["premise1", "premise2", ...],
    "conclusions": ["conclusion1", "conclusion2", ...],
    "logical_gaps": ["gap1", "gap2", ...],
    "unsupported_claims": ["claim1", "claim2", ...],
    "strength_score": <integer between 0 and 10>
}}

Response:
{text}
"""
        result = await processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        cleaned_content = processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse logical flow analysis response.")
            return {}

    @staticmethod
    async def identify_assumptions(processor, text: str) -> Dict:
        prompt = f"""Identify all assumptions in this response:
1. Explicit assumptions that are stated
2. Implicit assumptions that are relied upon
3. Critical dependencies
4. Domain-specific presumptions

Return your analysis as a JSON object with exactly this structure:
{{
    "explicit_assumptions": ["assumption1", "assumption2", ...],
    "implicit_assumptions": ["assumption1", "assumption2", ...],
    "dependencies": ["dependency1", "dependency2", ...],
    "domain_presumptions": ["presumption1", "presumption2", ...],
    "criticality_scores": {{
        "assumption1": <score 0-10>,
        "assumption2": <score 0-10>,
        ...
    }}
}}

Response:
{text}
"""
        result = await processor._get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        cleaned_content = processor._extract_json_from_content(result['content'])
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse assumptions identification response.")
            return {
                "explicit_assumptions": [],
                "implicit_assumptions": [],
                "dependencies": [],
                "domain_presumptions": [],
                "criticality_scores": {}
            }

# ---------------------------------------------------------------------------------------
# RetryHandler, VerificationLogger, etc.
# ---------------------------------------------------------------------------------------
class RetryHandler:
    def __init__(self, max_retries=3, base_delay=1):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        raise last_error

class VerificationLogger:
    def __init__(self, logger):
        self.logger = logger
        self.verification_id = str(uuid.uuid4())
        self.start_time = time.time()

    def log_verification_step(self, step: str, details: Dict):
        elapsed = time.time() - self.start_time
        log_entry = {
            "verification_id": self.verification_id,
            "step": step,
            "elapsed_time": elapsed,
            "details": details
        }
        self.logger.debug(f"Verification step: {json.dumps(log_entry, indent=2)}")

    def log_verification_result(self, result: Dict):
        elapsed = time.time() - self.start_time
        log_entry = {
            "verification_id": self.verification_id,
            "total_time": elapsed,
            "result": result
        }
        self.logger.info("Verification complete (debug only).")
        self.logger.debug(f"Verification complete: {json.dumps(log_entry, indent=2)}")


class VerificationResultFormatter:
    def format_verification_report(self, results: Dict) -> Panel:
        verification_content = Group(
            Text("🔍 Verification Results", style="bold blue"),
            self._format_claim_verification(results.get("claim_verification", [])),
            self._format_logical_analysis(results.get("logical_analysis", {})),
            self._format_mathematical_validation(results.get("mathematical_validation", [])),
            self._format_assumptions(results.get("assumptions", {})),
            Text(f"\nOverall Verification Score: {results.get('verification_score', 0):.2f}", style="bold green")
        )

        return Panel(
            verification_content,
            title="Verification Report",
            border_style="blue"
        )

    def _format_claim_verification(self, claims: List[Dict]) -> Group:
        content = [Text("Claims Verification:", style="bold")]
        for i, claim in enumerate(claims, 1):
            score = (
                claim.get("validity", 0)
                + claim.get("correctness", 0)
                + claim.get("completeness", 0)
            ) / 3
            content.append(Text(f"{i}. Score: {score:.1f}/10", style="cyan"))
            if claim.get("issues"):
                content.append(Text("   Issues:", style="yellow"))
                for issue in claim["issues"]:
                    content.append(Text(f"   • {issue}", style="yellow"))
            if claim.get("suggestions"):
                content.append(Text("   Suggestions:", style="green"))
                for suggestion in claim["suggestions"]:
                    content.append(Text(f"   • {suggestion}", style="green"))
        return Group(*content)

    def _format_logical_analysis(self, analysis: Dict) -> Group:
        content = [Text("Logical Analysis:", style="bold")]
        if analysis.get("logical_gaps"):
            content.append(Text("Logical Gaps:", style="yellow"))
            for gap in analysis["logical_gaps"]:
                content.append(Text(f"• {gap}", style="yellow"))
        if analysis.get("unsupported_claims"):
            content.append(Text("Unsupported Claims:", style="yellow"))
            for claim in analysis["unsupported_claims"]:
                content.append(Text(f"• {claim}", style="yellow"))
        score = analysis.get("strength_score", 0)
        content.append(Text(f"Strength Score: {score:.1f}/10", style="cyan"))
        return Group(*content)

    def _format_mathematical_validation(self, steps: List[Dict]) -> Group:
        content = [Text("Mathematical Validation:", style="bold")]
        for i, step in enumerate(steps, 1):
            content.append(Text(f"{i}. {step.get('step', '')}", style="cyan"))
            if step.get("justification"):
                content.append(Text(f"   Justification: {step.get('justification')}", style="dim"))
        return Group(*content)

    def _format_assumptions(self, assumptions: Dict) -> Group:
        content = [Text("Assumptions Analysis:", style="bold")]
        if assumptions.get("explicit_assumptions"):
            content.append(Text("Explicit Assumptions:", style="yellow"))
            for assumption in assumptions["explicit_assumptions"]:
                content.append(Text(f"• {assumption}", style="yellow"))
        if assumptions.get("implicit_assumptions"):
            content.append(Text("Implicit Assumptions:", style="yellow"))
            for assumption in assumptions["implicit_assumptions"]:
                content.append(Text(f"• {assumption}", style="yellow"))
        return Group(*content)


class VerificationMetrics:
    def __init__(self):
        self.metrics_file = "verification_metrics.csv"
        self.metrics = []

    def record_verification(self, results: Dict):
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "verification_score": results.get("verification_score", 0),
            "num_claims": len(results.get("claim_verification", [])),
            "num_logical_gaps": len(results.get("logical_analysis", {}).get("logical_gaps", [])),
            "num_unsupported_claims": len(results.get("logical_analysis", {}).get("unsupported_claims", [])),
            "num_assumptions": len(results.get("assumptions", {}).get("implicit_assumptions", [])),
            "processing_time": results.get("processing_time", 0)
        }
        self.metrics.append(metrics)
        self._save_metrics()

    def _save_metrics(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metrics_file, index=False)

    def get_metrics_summary(self) -> Dict:
        if not self.metrics:
            return {}
        df = pd.DataFrame(self.metrics)
        return {
            "avg_verification_score": df["verification_score"].mean(),
            "avg_processing_time": df["processing_time"].mean(),
            "total_verifications": len(df),
            "success_rate": (df["verification_score"] >= 7.0).mean() if not df.empty else 0
        }


# ---------------------------------------------------------------------------------------
# VerificationOrchestrator
# ---------------------------------------------------------------------------------------
class VerificationOrchestrator:
    def __init__(self, processor):
        self.processor = processor
        self.pipeline = VerificationPipeline(processor)
        self.analyzer = ResponseAnalyzer()
        self.retry_handler = RetryHandler()
        self.logger = VerificationLogger(logger)
        self.formatter = VerificationResultFormatter()
        self.metrics = VerificationMetrics()

    async def verify_response(self, response: str) -> Dict:
        try:
            self.logger.log_verification_step("start", {"response_length": len(response)})
            results = {}

            async def run_verification(name, func, *args):
                try:
                    out = await self.retry_handler.execute(func, *args)
                    self.logger.log_verification_step(name, {"status": "success"})
                    return out
                except Exception as e:
                    self.logger.log_verification_step(name, {"status": "error", "error": str(e)})
                    raise Exception(f"Verification step {name} failed: {str(e)}")

            # 1) Extract claims
            claims = await run_verification("claims", self.pipeline.extract_claims, response)
            claim_verifications = []
            if claims:
                async with asyncio.TaskGroup() as tg:
                    tasks = []
                    for c in claims:
                        task = tg.create_task(
                            run_verification(f"verify_claim_{uuid.uuid4()}", self.pipeline.verify_claim, c, response)
                        )
                        tasks.append((c, task))
                for c, t in tasks:
                    claim_verifications.append(t.result())

            results["claim_verification"] = claim_verifications

            # 2) Logical analysis
            logical_analysis = await run_verification(
                "logical_analysis", self.analyzer.check_logical_flow, self.processor, response
            )
            results["logical_analysis"] = logical_analysis

            # 3) Mathematical validation
            mathematical_validation = await run_verification(
                "mathematical_validation", self.analyzer.extract_mathematical_steps, self.processor, response
            )
            results["mathematical_validation"] = mathematical_validation

            # 4) Assumptions
            assumptions = await run_verification(
                "assumptions", self.analyzer.identify_assumptions, self.processor, response
            )
            results["assumptions"] = assumptions

            # 5) Corner cases
            corner_cases = await run_verification(
                "corner_cases", self.pipeline.validate_corner_cases, response
            )
            results["corner_cases"] = corner_cases

            # 6) Relationships
            relationships = await run_verification(
                "relationships", self.pipeline.check_relationships, response
            )
            results["relationships"] = relationships

            # 7) Score
            verification_score = self._calculate_verification_score(results)
            results["verification_score"] = verification_score

            self.metrics.record_verification(results)

            # Render a Rich Panel -> ASCII
            # Use a 'with' block so we don't call .close() manually.
            with Console(record=True, width=100) as temp_console:
                verification_panel = self.formatter.format_verification_report(results)
                temp_console.print(verification_panel)
                ascii_verification = temp_console.export_text()

            results["formatted_report"] = ascii_verification
            self.logger.log_verification_result(results)

            return results

        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return {"error": f"Verification failed: {str(e)}"}

    def _calculate_verification_score(self, results: Dict) -> float:
        weights = {
            "claims": 0.25,
            "logic": 0.25,
            "math": 0.2,
            "assumptions": 0.1,
            "corner_cases": 0.1,
            "relationships": 0.1
        }
        scores = {
            "claims": self._avg_claim_scores(results.get("claim_verification", [])),
            "logic": results.get("logical_analysis", {}).get("strength_score", 0),
            "math": self._calc_math_score(results.get("mathematical_validation", [])),
            "assumptions": 10 - len(results.get("assumptions", {}).get("implicit_assumptions", [])),
            "corner_cases": 10 - len(results.get("corner_cases", {}).get("potential_issues", [])),
            "relationships": 10 - len(results.get("relationships", {}).get("contradictions", []))
        }
        for k in scores:
            scores[k] = max(0, min(scores[k], 10))

        return sum(scores[k] * weights[k] for k in weights)

    def _avg_claim_scores(self, claim_verifications: List[Dict]) -> float:
        if not claim_verifications:
            return 0
        sc = []
        for v in claim_verifications:
            avg = (
                v.get("validity", 0)
                + v.get("correctness", 0)
                + v.get("completeness", 0)
            ) / 3
            sc.append(avg)
        return sum(sc) / len(sc) if sc else 0

    def _calc_math_score(self, math_steps: List[Dict]) -> float:
        if not math_steps:
            return 0
        partial_scores = []
        for step in math_steps:
            if step.get("justification"):
                partial_scores.append(10)
            else:
                partial_scores.append(5)
        return sum(partial_scores) / len(partial_scores)

    # Called by aggregator to unify multiple responses
    async def _evaluate_general_responses(self, responses: List[Dict]) -> Dict:
        if not responses:
            return {"response": "", "score": 0, "type": "general", "tokens": 0}

        # Convert to text
        all_texts = [r["text"] for r in responses]
        cross_validation = await self.pipeline.cross_validate(all_texts)

        synthesis_prompt = f"""We have multiple responses with detailed verification results:

Responses:
{json.dumps(all_texts, indent=2)}

Cross-validation Results:
{json.dumps(cross_validation, indent=2)}

Create an improved final answer that:
1. Incorporates the strongest elements from all responses
2. Addresses all verification issues identified
3. Ensures mathematical correctness
4. Maintains logical consistency
5. Handles edge cases
6. Explicitly states key assumptions
7. Provides clear justification for each step

Return only the final combined text.
"""
        try:
            # We'll note the tokens before:
            start_tokens = self.processor.total_tokens
            synthesis = await self.processor._get_completion(synthesis_prompt, temperature=0.3)
            end_tokens = self.processor.total_tokens
            # the difference spent here
            spent_synthesis = end_tokens - start_tokens

            synthesis_text = synthesis["content"]
            # Now verify the final
            before_ver = self.processor.total_tokens
            final_verification = await self.verify_response(synthesis_text)
            after_ver = self.processor.total_tokens
            spent_ver = after_ver - before_ver

            return {
                "response": synthesis_text,
                "score": final_verification.get("verification_score", 0),
                "type": "general",
                "verification": final_verification,
                # add the tokens from the aggregator steps (for the final answer + verification)
                "tokens": synthesis.get("token_usage", 0) + spent_synthesis + spent_ver
            }
        except Exception as e:
            logger.error(f"Aggregation and synthesis failed: {e}")
            return {"response": "", "score": 0, "type": "general", "tokens": 0}


# ---------------------------------------------------------------------------------------
# EnhancedOpenAIProcessor
# ---------------------------------------------------------------------------------------
class EnhancedOpenAIProcessor:
    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = ResponseCache()
        self.total_tokens = 0
        self.initial_budget = INITIAL_BUDGET
        self.max_budget = MAX_BUDGET
        self.current_best_score = 0
        self.budget_increase_threshold = BUDGET_INCREASE_THRESHOLD
        self.generation_config = GENERATION_CONFIG
        self.console = Console()

    def _calculate_dynamic_budget(self, current_score: float = 0) -> int:
        base_budget = self.initial_budget
        if 0 < current_score < self.budget_increase_threshold:
            quality_multiplier = 1 + (self.budget_increase_threshold - current_score)
            base_budget = int(base_budget * quality_multiplier)
        return min(max(base_budget, self.initial_budget), self.max_budget)

    # Step prints
    async def _classify_query_type(self, query: str) -> Dict[str, Any]:
        self.console.print("[bold magenta]Step: Classifying query[/bold magenta]")
        prompt = f"""Analyze the following query's domain or category in general terms (like 'math', 'creative', 'data', etc.).
Return your response as a JSON object with exactly this structure:
{{
    "category": "category_name",
    "confidence": 0.0
}}

Query: {query}
"""
        try:
            response = await self._get_completion(prompt, temperature=0.1, response_format={"type": "json_object"})
            cleaned = self._extract_json_from_content(response['content'])
            data = json.loads(cleaned)
            return {
                "category": data.get("category", "generic"),
                "confidence": data.get("confidence", 0.5)
            }
        except Exception as e:
            logger.warning(f"Query classification minimal step failed: {e}")
            return {"category": "generic", "confidence": 0.5}

    async def _understand_goal(self, problem_description: str) -> str:
        self.console.print("[bold magenta]Step: Understanding goal[/bold magenta]")
        prompt = f"""Summarize the core goal of the following query in one concise sentence.
Return your response as a JSON object with exactly this structure:
{{
    "goal": "concise one-sentence summary of the goal"
}}

Query:
{problem_description}"""
        try:
            response_data = await self._get_completion(
                prompt,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            cleaned_content = self._extract_json_from_content(response_data["content"])
            data = json.loads(cleaned_content)
            return data["goal"].strip()
        except Exception as e:
            logger.error(f"_understand_goal failed: {e}")
            return ""

    async def _decompose_problem(self, problem_details: str) -> List[str]:
        self.console.print("[bold magenta]Step: Decomposing problem[/bold magenta]")
        prompt = f"""Break down the following query into its main functional components or sub-problems. 
Return your response as a JSON array with exactly this structure:
[
    "component1",
    "component2",
    ...
]

Query:
{problem_details}"""
        try:
            response_data = await self._get_completion(
                prompt,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            cleaned_content = self._extract_json_from_content(response_data['content'])
            return json.loads(cleaned_content)
        except Exception as e:
            logger.error(f"_decompose_problem failed: {e}")
            return []

    async def _extract_and_analyze_information(self, input_data: str) -> Dict:
        self.console.print("[bold magenta]Step: Extracting info[/bold magenta]")
        prompt = f"""Analyze the following query and extract key features, potential complexities, and any specific information 
that needs careful consideration. Return your analysis as a JSON object with exactly this structure:
{{
    "key_features": ["feature1", "feature2", ...],
    "complexities": ["complexity1", "complexity2", ...],
    "considerations": ["consideration1", "consideration2", ...],
    "analysis_summary": "brief summary text"
}}

Query:
{input_data}"""
        try:
            response_data = await self._get_completion(
                prompt,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            cleaned_content = self._extract_json_from_content(response_data["content"])
            return json.loads(cleaned_content)
        except Exception as e:
            logger.error(f"_extract_and_analyze_information failed: {e}")
            return {}

    async def _recognize_patterns(self, analyzed_information: Dict) -> List[str]:
        self.console.print("[bold magenta]Step: Recognizing patterns[/bold magenta]")
        prompt = f"""Based on the following analysis, identify relevant problem-solving strategies, common approaches, or any known patterns 
that could be helpful in addressing this query. Return your response as a JSON array with exactly this structure:
[
    "pattern1",
    "pattern2",
    ...
]

Analysis:
{json.dumps(analyzed_information, indent=2)}"""
        try:
            response_data = await self._get_completion(
                prompt,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            cleaned_content = self._extract_json_from_content(response_data["content"])
            return json.loads(cleaned_content)
        except Exception as e:
            logger.error(f"_recognize_patterns failed: {e}")
            return []

    async def _generate_hypotheses(self, analysis: Dict, patterns: List[str]) -> List[str]:
        self.console.print("[bold magenta]Step: Generating hypotheses[/bold magenta]")
        patterns_str = "\n".join(f"- {p}" for p in patterns)
        prompt = f"""Based on the provided analysis and recognized patterns, suggest potential strategies or approaches for effectively 
addressing the query. Return your response as a JSON array with exactly this structure:
[
    "strategy1",
    "strategy2",
    ...
]

Analysis: {json.dumps(analysis, indent=2)}
Recognized Patterns:
{patterns_str}
"""
        try:
            response_data = await self._get_completion(
                prompt,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            cleaned_content = self._extract_json_from_content(response_data['content'])
            return json.loads(cleaned_content)
        except Exception as e:
            logger.error(f"_generate_hypotheses failed: {e}")
            return []

    async def _prioritize_suggestions(self, hypotheses: List[str]) -> List[Tuple[str, Dict]]:
        self.console.print("[bold magenta]Step: Prioritizing suggestions[/bold magenta]")
        if not hypotheses:
            return []
        bullet_hypotheses = "\n".join(f"- {h}" for h in hypotheses)
        prompt = f"""Rank the following suggested approaches based on their potential impact on improving the response quality 
and their feasibility of implementation. Provide a score (0-10) for both 'impact' and 'feasibility' for each suggestion. 

Suggested Approaches:
{bullet_hypotheses}

Return your analysis as a JSON object where each key is a suggestion and its value is an object containing 'impact' and 'feasibility' scores.
Example format:
{{
    "suggestion1": {{ "impact": 8, "feasibility": 7 }},
    "suggestion2": {{ "impact": 6, "feasibility": 9 }}
}}
"""
        try:
            response_data = await self._get_completion(
                prompt, response_format={"type": "json_object"}, temperature=0.3
            )
            cleaned_content = self._extract_json_from_content(response_data["content"])
            ranking_dict = json.loads(cleaned_content)
            ranked_items = []
            for suggestion, scores in ranking_dict.items():
                impact = float(scores.get("impact", 0))
                feasibility = float(scores.get("feasibility", 0))
                ranked_items.append((suggestion, {"impact": impact, "feasibility": feasibility}))
            ranked_items.sort(key=lambda x: x[1]["impact"] + x[1]["feasibility"], reverse=True)
            return ranked_items
        except Exception as e:
            logger.error(f"_prioritize_suggestions failed: {e}")
            return []

    async def _structure_reasoning(self, prioritized_suggestions: List[Tuple[str, Dict]]) -> str:
        self.console.print("[bold magenta]Step: Structuring reasoning[/bold magenta]")
        if not prioritized_suggestions:
            return "No suggestions to structure."
        suggestions_text = "\n".join(
            f"- {s[0]} (impact={s[1]['impact']}, feasibility={s[1]['feasibility']})"
            for s in prioritized_suggestions
        )
        prompt = f"""Based on the following prioritized suggestions (with impact and feasibility scores), 
formulate a coherent and actionable plan or strategy for processing the user's query. 
Explain the reasoning behind the chosen approach:

{suggestions_text}
"""
        try:
            response_data = await self._get_completion(prompt)
            return response_data["content"].strip()
        except Exception as e:
            logger.error(f"_structure_reasoning failed: {e}")
            return ""

    async def _get_completion(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        response_format: Dict = None
    ) -> Dict:
        max_retries = 3
        retry_delay = 1

        logger.debug("Sending prompt to model:")
        logger.debug(prompt)
        logger.debug(f"Parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, response_format={response_format}")

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature if temperature is not None else self.generation_config["temperature"],
                    max_tokens=max_tokens if max_tokens is not None else self.generation_config["max_tokens"],
                    top_p=top_p if top_p is not None else self.generation_config["top_p"],
                )
                logger.debug("Raw model response:")
                logger.debug(json.dumps(response.model_dump(), indent=2))

                content = response.choices[0].message.content.strip()
                token_usage = response.usage.total_tokens

                logger.debug("Extracted content:")
                logger.debug(content)
                logger.debug(f"Token usage: {token_usage}")

                self.total_tokens += token_usage

                clean_content = self._extract_json_from_content(content)
                return {
                    'content': clean_content,
                    'token_usage': token_usage
                }

            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise

    def _extract_json_from_content(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```[a-zA-Z]+\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
        return content.strip()

    async def _generate_sequential(
        self,
        query: str,
        prev_responses: List[Dict],
        strategy: ProcessingStrategy
    ) -> Dict:
        start_time = time.time()
        self.console.print("[bold magenta]Step: Generating sequential responses[/bold magenta]")
        max_history = min(len(prev_responses), strategy.max_revision_steps)
        if max_history > 0:
            context = "\n\n".join([
                f"Previous attempt {i+1}:\n{resp['text']}"
                for i, resp in enumerate(prev_responses[-max_history:])
            ])
            prompt = f"""Original query:
{query}

Previous attempts:
{context}

Based on the previous attempts, refine and improve the solution, ensuring correctness, clarity, and completeness.
Return the improved response.
"""
        else:
            prompt = f"""Original query:
{query}

Provide a thorough solution, ensuring correctness, clarity, and completeness.
"""

        response_data = await self._get_completion(
            prompt,
            temperature=strategy.temperature,
            max_tokens=strategy.max_tokens,
            top_p=strategy.top_p
        )

        duration = time.time() - start_time
        return {
            'text': response_data['content'],
            'type': 'Sequential',
            'duration': duration,
            'tokens': response_data['token_usage']
        }

    async def _generate_parallel(
        self,
        query: str,
        strategy: ProcessingStrategy
    ) -> List[Dict]:
        self.console.print("[bold magenta]Step: Generating parallel responses[/bold magenta]")
        base_temp = strategy.temperature
        step = 0.1
        half = strategy.parallel_samples // 2
        temp_variations = []
        for i in range(strategy.parallel_samples):
            offset = (i - half) * step
            varied_temp = max(0.0, min(2.0, base_temp + offset))
            temp_variations.append(varied_temp)

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for i, t in enumerate(temp_variations, start=1):
                task = tg.create_task(self._generate_parallel_response(
                    query, strategy, i, varied_temp=t
                ))
                tasks.append(task)

        results = [t.result() for t in tasks if t.result() is not None]
        return results

    async def _generate_parallel_response(
        self,
        query: str,
        strategy: ProcessingStrategy,
        index: int,
        varied_temp: float
    ) -> Dict:
        start_time = time.time()
        domain_prompt = f"""(Parallel Attempt #{index}, temperature={varied_temp:.2f})
User query:
{query}

Generate a response focusing on correctness, clarity, completeness.
Return your best solution.
"""
        response_data = await self._get_completion(
            domain_prompt,
            temperature=varied_temp,
            max_tokens=strategy.max_tokens,
            top_p=strategy.top_p
        )
        duration = time.time() - start_time
        return {
            'text': response_data['content'],
            'type': 'Parallel',
            'duration': duration,
            'tokens': response_data['token_usage']
        }

    async def _aggregate_responses(self, responses: List[Dict]) -> Dict:
        logger.info(f"Starting response aggregation. Number of responses to aggregate: {len(responses)}")
        self.console.print("[bold magenta]Step: Aggregating responses[/bold magenta]")
        if not responses:
            logger.warning("No responses to aggregate")
            return None

        orchestrator = VerificationOrchestrator(self)
        # We measure tokens before aggregator:
        start_tokens = self.total_tokens
        best_response = await orchestrator._evaluate_general_responses(responses)
        aggregator_spent = self.total_tokens - start_tokens

        if best_response and best_response.get("response"):
            # If final_check was done inside orchestrator, it includes that usage
            # So let's incorporate aggregator_spent into best_response's 'tokens'
            if "tokens" in best_response:
                best_response["tokens"] += aggregator_spent

        return best_response

    def format_response_table(self, responses: List[Dict], start_time: float) -> Table:
        table = Table(title="Generated Responses", show_header=True, header_style="bold magenta")
        table.add_column("№", style="dim")
        table.add_column("Type")
        table.add_column("Response Preview", width=60)
        table.add_column("Time", style="dim")

        for i, resp in enumerate(responses, 1):
            preview = resp['text'][:200]
            if len(resp['text']) > 200:
                preview += "..."
            table.add_row(str(i), resp['type'], Text(preview, no_wrap=False), f"{resp['duration']:.2f}s")

        return table

    def format_result_panel(self, result: Dict, duration: float) -> Panel:
        if "error" in result:
            return Panel(
                "\n".join(["❌ Error", result["error"]]),
                title="Error",
                border_style="red"
            )

        strategy_info = result.get('strategy', {})
        budget_used = result.get('total_budget_used', INITIAL_BUDGET)
        budget_percent = (budget_used / MAX_BUDGET) * 100

        total_calls = len(result.get('all_responses', [])) + 1
        total_tokens = result.get("total_tokens", 0)
        avg_tokens_per_call = total_tokens / total_calls if total_calls else 0

        analysis_content = Group(
            Text("📊 Analysis", style="bold blue"),
            Text(f"Detected Category: {result.get('query_type', 'generic')}"),
            Text(f"Strategy: Seq={strategy_info.get('sequential_samples', 2)}, Par={strategy_info.get('parallel_samples', 2)}"),
            Text(f"Budget Used: {budget_used}/{MAX_BUDGET} ({budget_percent:.1f}%)"),
            Text(f"Total Duration: {duration:.2f}s")
        )
        analysis_panel = Panel(analysis_content, border_style="blue", width=60)

        token_content = Group(
            Text("🔢 Token Usage", style="bold cyan"),
            Text(f"Avg Tokens per Call: {avg_tokens_per_call:.0f}"),
            Text(f"Total API Calls: {total_calls}"),
            Text(f"Total Tokens Used: {total_tokens}")
        )
        token_panel = Panel(token_content, border_style="cyan", width=60)

        best_response = result.get("best_response")
        if not best_response:
            best_resp_panel = Panel("No valid best response was selected.", border_style="red")
            verification_panel = Panel("No verification report available.", border_style="red")
        else:
            best_resp_panel = Panel(
                Group(
                    Text("🎯 Best Response", style="bold green"),
                    Text(best_response['response'], style="yellow"),
                    Text(f"\nScore: {best_response.get('score', 0):.2f}"),
                    Text(f"Type: {best_response.get('type', 'N/A')}")
                ),
                border_style="green",
                width=120
            )
            verification_report = best_response.get("verification", {}).get("formatted_report")
            if verification_report:
                verification_panel = Panel(
                    Text(verification_report, justify="left"),
                    title="Verification Report",
                    border_style="blue"
                )
            else:
                verification_panel = Panel("No verification report available.", border_style="red")

        return Panel(
            Group(
                Columns([analysis_panel, token_panel]),
                best_resp_panel,
                verification_panel
            ),
            title="Final Results",
            border_style="bright_blue"
        )

    def store_usage_stats(self, query: str, result: Dict, csv_filename: str = "usage_stats.csv"):
        fieldnames = [
            "timestamp", "query", "query_type",
            "total_tokens_used", "processing_time", "best_score"
        ]
        row = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_type": result.get("query_type", "generic"),
            "total_tokens_used": result.get("total_tokens", 0),
            "processing_time": f"{result.get('processing_time', 0):.2f}",
            "best_score": result.get("best_response", {}).get("score", 0)
        }
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    async def process_query(self, query: str, total_budget: int = None) -> Dict:
        start_time = time.time()
        total_tokens = 0

        try:
            classification = await self._classify_query_type(query)
            goal = await self._understand_goal(query)
            components = await self._decompose_problem(query)
            analysis = await self._extract_and_analyze_information(query)
            patterns = await self._recognize_patterns(analysis)
            hypotheses = await self._generate_hypotheses(analysis, patterns)
            prioritized_suggestions = await self._prioritize_suggestions(hypotheses)
            reasoning_strategy = await self._structure_reasoning(prioritized_suggestions)

            logger.info(f"Reasoning Strategy:\n{reasoning_strategy}")

            if total_budget is None:
                total_budget = self._calculate_dynamic_budget()

            strategy = ProcessingStrategy(
                sequential_samples=2,
                parallel_samples=2,
                temperature=GENERATION_CONFIG["temperature"],
                max_tokens=GENERATION_CONFIG["max_tokens"],
                top_p=GENERATION_CONFIG["top_p"],
            )

            all_responses = []

            # Generate sequential
            seq_resps = []
            if strategy.sequential_samples > 0:
                for _ in range(strategy.sequential_samples):
                    resp = await self._generate_sequential(query, seq_resps, strategy)
                    seq_resps.append(resp)
                    all_responses.append(resp)
                    total_tokens += resp['tokens']
                    self.total_tokens = total_tokens

            # Generate parallel
            if strategy.parallel_samples > 0:
                par_resps = await self._generate_parallel(query, strategy)
                for r in par_resps:
                    all_responses.append(r)
                    total_tokens += r['tokens']
                    self.total_tokens = total_tokens

            # Aggregate
            best_response = await self._aggregate_responses(all_responses)
            if best_response:
                total_tokens += best_response.get("tokens", 0)
                self.total_tokens = total_tokens

            duration = time.time() - start_time
            result = {
                "query_type": classification.get("category", "generic"),
                "best_response": best_response,
                "all_responses": all_responses,
                "processing_time": duration,
                "total_budget_used": total_budget,
                "total_tokens": self.total_tokens,  # Store final sum
                "strategy": {
                    "sequential_samples": strategy.sequential_samples,
                    "parallel_samples": strategy.parallel_samples,
                    "temperature": strategy.temperature,
                    "top_p": strategy.top_p
                },
                "reasoning_goal": goal,
                "reasoning_components": components,
                "reasoning_analysis": analysis,
                "reasoning_patterns": patterns,
                "reasoning_prioritized": prioritized_suggestions,
                "reasoning_strategy": reasoning_strategy
            }
            return result

        except Exception as e:
            error_msg = f"Error in process_query: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "processing_time": time.time() - start_time,
                "total_tokens": self.total_tokens
            }


# ---------------------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------------------
async def main():
    console = Console()
    logger.info("Starting new run")

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        processor = EnhancedOpenAIProcessor(api_key)

        default_queries = [
            ("Analytical", "Find the volume of a cone with radius 5 and height 12."),
            ("Creative", "Write a short story about a cat discovering a magical portal."),
            ("Informational", "Explain how photosynthesis works in simple terms."),
            ("Open-ended", "What are your thoughts on the future of artificial intelligence?"),
            ("Procedural", "How do I make a perfect omelet?"),
            ("Complex Math",
             "Given a sphere inscribed in a cylinder, which is inscribed in a cone with height h and base radius r, "
             "find the ratio of the volumes of the cone to cylinder to sphere in terms of h and r, and prove that "
             "this ratio is constant regardless of the dimensions. Then find the minimum value of h/r that makes "
             "this configuration possible.")
        ]

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

        for i, (category, q) in enumerate(default_queries, 1):
            console.print(f"[cyan]{i}.[/cyan] [bold]{category}[/bold]")
            console.print(f"   {q[:100]}..." if len(q) > 100 else f"   {q}")
            console.print()

        console.print("[cyan]C.[/cyan] [bold]Custom Query[/bold]\n")

        while True:
            choice = console.input("[yellow]Enter your choice (1-6 or C): [/yellow]").strip().upper()
            if choice == 'C':
                query = console.input("\n[yellow]Enter your query: [/yellow]").strip()
                if query:
                    break
            elif choice.isdigit() and 1 <= int(choice) <= len(default_queries):
                query = default_queries[int(choice) - 1][1]
                break
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")

        console.print("\n[bold green]Processing your query...[/bold green]\n")

        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task("Analyzing your query...", start=False)

            async def track_progress():
                progress.start_task(task_id)
                while not progress.finished:
                    progress.update(
                        task_id,
                        description=f"Analyzing your query... (Tokens used so far: {processor.total_tokens})"
                    )
                    await asyncio.sleep(0.2)

            tracker = asyncio.create_task(track_progress())

            console.print(Panel(query, title="Question", border_style="yellow"))
            result = await processor.process_query(query)
            duration = result.get("processing_time", 0)

            progress.update(task_id, completed=100)
            progress.stop()

            tracker.cancel()
            try:
                await tracker
            except asyncio.CancelledError:
                pass

        console.print("\n")
        responses_table = processor.format_response_table(result.get("all_responses", []), time.time() - duration)
        console.print(responses_table)

        console.print("\n")
        result_panel = processor.format_result_panel(result, duration)
        console.print(result_panel)

        processor.store_usage_stats(query, result)

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