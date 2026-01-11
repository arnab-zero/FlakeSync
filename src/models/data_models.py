"""Data models for the flaky test detection and repair system."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class FlakinessCategory(str, Enum):
    """Flakiness categories."""
    ASYNC_WAIT = "async wait"
    UNORDERED_COLLECTIONS = "unordered collections"
    CONCURRENCY = "concurrency"
    TIME = "time"
    TEST_ORDER_DEPENDENCY = "test order dependency"


@dataclass
class TestMethod:
    """Represents a test method extracted from Java source code."""
    name: str
    body: str
    annotations: List[str]
    start_line: int
    end_line: int
    file_path: str
    signature: str
    
    def __post_init__(self):
        """Validate test method data."""
        if not self.name:
            raise ValueError("Test method name cannot be empty")
        if self.start_line < 0 or self.end_line < 0:
            raise ValueError("Line numbers must be non-negative")
        if self.start_line > self.end_line:
            raise ValueError("start_line must be <= end_line")


@dataclass
class DetectionResult:
    """Results from flakiness detection."""
    is_flaky: bool
    category: str
    confidence: float
    all_scores: Dict[str, float]
    method_name: str
    
    def __post_init__(self):
        """Validate detection result."""
        if self.confidence < 0.0 or self.confidence > 100.0:
            raise ValueError("Confidence must be between 0 and 100")
        
        # Validate category
        valid_categories = [cat.value for cat in FlakinessCategory]
        if self.category not in valid_categories:
            raise ValueError(f"Invalid category: {self.category}. Must be one of {valid_categories}")


@dataclass
class MethodInvocation:
    """Represents a method invocation within code."""
    method_name: str
    receiver: Optional[str]  # Object/class on which method is called
    arguments: List[str]
    line_number: int
    is_external: bool  # True if third-party library
    
    def __post_init__(self):
        """Validate method invocation."""
        if not self.method_name:
            raise ValueError("Method name cannot be empty")
        if self.line_number < 0:
            raise ValueError("Line number must be non-negative")


@dataclass
class MethodNode:
    """Represents a node in the call graph."""
    method_id: str
    method_name: str
    file_path: str
    is_external: bool
    method_body: Optional[str] = None
    
    def __post_init__(self):
        """Validate method node."""
        if not self.method_id:
            raise ValueError("Method ID cannot be empty")


@dataclass
class CallGraph:
    """Represents a call graph for interprocedural analysis."""
    nodes: Dict[str, MethodNode]  # method_id -> MethodNode
    edges: List[Tuple[str, str]]  # (caller_id, callee_id)
    root: str  # Test method ID
    
    def __post_init__(self):
        """Validate call graph."""
        if self.root not in self.nodes:
            raise ValueError(f"Root node {self.root} not found in nodes")
        
        # Validate edges reference existing nodes
        for caller, callee in self.edges:
            if caller not in self.nodes:
                raise ValueError(f"Edge references non-existent caller: {caller}")
            if callee not in self.nodes:
                raise ValueError(f"Edge references non-existent callee: {callee}")
    
    def get_callees(self, method_id: str) -> List[str]:
        """Get all methods called by the given method."""
        return [callee for caller, callee in self.edges if caller == method_id]
    
    def get_callers(self, method_id: str) -> List[str]:
        """Get all methods that call the given method."""
        return [caller for caller, callee in self.edges if callee == method_id]
    
    def has_cycle(self) -> bool:
        """Check if the call graph contains cycles."""
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for callee in self.get_callees(node_id):
                if callee not in visited:
                    if has_cycle_util(callee):
                        return True
                elif callee in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle_util(node_id):
                    return True
        return False


@dataclass
class MethodDefinition:
    """Represents a method definition resolved from invocation."""
    method_name: str
    file_path: str
    body: str
    start_line: int
    end_line: int
    is_external: bool


@dataclass
class AnalysisResult:
    """Results from interprocedural analysis."""
    test_method: TestMethod
    call_graph: CallGraph
    flaky_methods: List[Tuple[MethodDefinition, DetectionResult]]
    external_dependencies: List[MethodInvocation]
    aggregated_flakiness: DetectionResult  # Combined prediction
    
    def get_all_flaky_categories(self) -> List[str]:
        """Get all unique flakiness categories found."""
        categories = {self.aggregated_flakiness.category}
        for _, result in self.flaky_methods:
            categories.add(result.category)
        return list(categories)


@dataclass
class FlakyLine:
    """Represents a specific flaky line within a method."""
    line_number: int
    code: str
    impact_score: float  # 0-1
    reason: str
    method_name: str
    file_path: str
    
    def __post_init__(self):
        """Validate flaky line."""
        if self.line_number < 0:
            raise ValueError("Line number must be non-negative")
        if self.impact_score < 0.0 or self.impact_score > 1.0:
            raise ValueError("Impact score must be between 0 and 1")


@dataclass
class CodeLocation:
    """Represents a specific location in code."""
    file_path: str
    line_number: int
    code_snippet: str
    method_name: str
    location_type: str  # "critical" or "barrier"
    
    def __post_init__(self):
        """Validate code location."""
        if self.line_number < 0:
            raise ValueError("Line number must be non-negative")
        if self.location_type not in ["critical", "barrier"]:
            raise ValueError("location_type must be 'critical' or 'barrier'")


@dataclass
class CriticalBarrierPoints:
    """Critical and barrier points for async flaky tests."""
    critical_points: List[CodeLocation]
    barrier_points: List[CodeLocation]
    dependencies: Dict[CodeLocation, List[CodeLocation]]  # critical -> barriers
    in_developer_code: bool
    
    def __post_init__(self):
        """Validate critical barrier points."""
        # Validate all critical points have "critical" type
        for point in self.critical_points:
            if point.location_type != "critical":
                raise ValueError(f"Critical point has wrong type: {point.location_type}")
        
        # Validate all barrier points have "barrier" type
        for point in self.barrier_points:
            if point.location_type != "barrier":
                raise ValueError(f"Barrier point has wrong type: {point.location_type}")


@dataclass
class SyncPattern:
    """Synchronization pattern template."""
    name: str
    description: str
    template: str  # Code template with placeholders
    example: str  # Real-world example
    applicable_categories: List[str]  # Flakiness categories this pattern fixes
    
    def __post_init__(self):
        """Validate sync pattern."""
        if not self.name:
            raise ValueError("Pattern name cannot be empty")
        if not self.template:
            raise ValueError("Pattern template cannot be empty")
        
        # Validate applicable categories
        valid_categories = [cat.value for cat in FlakinessCategory]
        for category in self.applicable_categories:
            if category not in valid_categories:
                raise ValueError(f"Invalid category: {category}")


@dataclass
class ValidationResult:
    """Results from patch validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a validation warning."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


@dataclass
class Patch:
    """Represents a generated patch for fixing flaky tests."""
    diff: str  # Unified diff format
    description: str
    confidence: float  # 0-1
    pattern_used: str  # e.g., "volatile_flag_spin_wait"
    modified_files: List[str]
    validation_result: Optional[ValidationResult] = None
    
    def __post_init__(self):
        """Validate patch."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if not self.diff:
            raise ValueError("Patch diff cannot be empty")
    
    def is_valid(self) -> bool:
        """Check if patch is valid."""
        if self.validation_result is None:
            return False
        return self.validation_result.is_valid
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        if self.validation_result is None:
            return ["Patch not validated"]
        return self.validation_result.errors


# Utility functions for data models

def create_test_method_from_dict(data: Dict) -> TestMethod:
    """Create TestMethod from dictionary."""
    return TestMethod(
        name=data['name'],
        body=data['body'],
        annotations=data.get('annotations', []),
        start_line=data['start_line'],
        end_line=data['end_line'],
        file_path=data['file_path'],
        signature=data.get('signature', '')
    )


def create_detection_result_from_dict(data: Dict) -> DetectionResult:
    """Create DetectionResult from dictionary."""
    return DetectionResult(
        is_flaky=data['is_flaky'],
        category=data['category'],
        confidence=data['confidence'],
        all_scores=data.get('all_scores', {}),
        method_name=data['method_name']
    )


def serialize_patch(patch: Patch) -> Dict:
    """Serialize Patch to dictionary for JSON output."""
    return {
        'diff': patch.diff,
        'description': patch.description,
        'confidence': patch.confidence,
        'pattern_used': patch.pattern_used,
        'modified_files': patch.modified_files,
        'is_valid': patch.is_valid(),
        'validation_errors': patch.get_validation_errors() if not patch.is_valid() else []
    }


def serialize_analysis_result(result: AnalysisResult) -> Dict:
    """Serialize AnalysisResult to dictionary for JSON output."""
    return {
        'test_method': {
            'name': result.test_method.name,
            'file_path': result.test_method.file_path,
            'start_line': result.test_method.start_line,
            'end_line': result.test_method.end_line
        },
        'aggregated_flakiness': {
            'is_flaky': result.aggregated_flakiness.is_flaky,
            'category': result.aggregated_flakiness.category,
            'confidence': result.aggregated_flakiness.confidence
        },
        'flaky_methods_count': len(result.flaky_methods),
        'external_dependencies_count': len(result.external_dependencies),
        'all_categories': result.get_all_flaky_categories()
    }
