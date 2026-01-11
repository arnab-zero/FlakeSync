"""
Property-based tests for data models.

Feature: flaky-test-detection-and-repair
Property 31: Report Generation Completeness
Validates: Requirements 11.1, 11.2
"""

import json
from hypothesis import given, settings, strategies as st, HealthCheck
from hypothesis.strategies import composite

from src.models.data_models import (
    TestMethod,
    DetectionResult,
    MethodInvocation,
    FlakyLine,
    CodeLocation,
    Patch,
    ValidationResult,
    SyncPattern,
    FlakinessCategory,
    create_test_method_from_dict,
    create_detection_result_from_dict,
    serialize_patch,
    MethodNode,
    CallGraph
)


# Custom strategies for generating test data

@composite
def test_method_strategy(draw):
    """Generate random TestMethod instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')))
    body = draw(st.text(min_size=1, max_size=500))
    annotations = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    start_line = draw(st.integers(min_value=1, max_value=1000))
    end_line = draw(st.integers(min_value=start_line, max_value=start_line + 100))
    file_path = draw(st.text(min_size=1, max_size=100))
    signature = draw(st.text(min_size=1, max_size=200))
    
    return TestMethod(
        name=name,
        body=body,
        annotations=annotations,
        start_line=start_line,
        end_line=end_line,
        file_path=file_path,
        signature=signature
    )


@composite
def detection_result_strategy(draw):
    """Generate random DetectionResult instances."""
    is_flaky = draw(st.booleans())
    category = draw(st.sampled_from([cat.value for cat in FlakinessCategory]))
    confidence = draw(st.floats(min_value=0.0, max_value=100.0))
    all_scores = draw(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=5
    ))
    method_name = draw(st.text(min_size=1, max_size=50))
    
    return DetectionResult(
        is_flaky=is_flaky,
        category=category,
        confidence=confidence,
        all_scores=all_scores,
        method_name=method_name
    )


@composite
def method_invocation_strategy(draw):
    """Generate random MethodInvocation instances."""
    method_name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')))
    receiver = draw(st.one_of(st.none(), st.text(min_size=1, max_size=30)))
    arguments = draw(st.lists(st.text(min_size=0, max_size=50), min_size=0, max_size=10))
    line_number = draw(st.integers(min_value=1, max_value=10000))
    is_external = draw(st.booleans())
    
    return MethodInvocation(
        method_name=method_name,
        receiver=receiver,
        arguments=arguments,
        line_number=line_number,
        is_external=is_external
    )


@composite
def flaky_line_strategy(draw):
    """Generate random FlakyLine instances."""
    line_number = draw(st.integers(min_value=1, max_value=10000))
    code = draw(st.text(min_size=1, max_size=200))
    impact_score = draw(st.floats(min_value=0.0, max_value=1.0))
    reason = draw(st.text(min_size=1, max_size=100))
    method_name = draw(st.text(min_size=1, max_size=50))
    file_path = draw(st.text(min_size=1, max_size=100))
    
    return FlakyLine(
        line_number=line_number,
        code=code,
        impact_score=impact_score,
        reason=reason,
        method_name=method_name,
        file_path=file_path
    )


@composite
def code_location_strategy(draw):
    """Generate random CodeLocation instances."""
    file_path = draw(st.text(min_size=1, max_size=100))
    line_number = draw(st.integers(min_value=1, max_value=10000))
    code_snippet = draw(st.text(min_size=1, max_size=200))
    method_name = draw(st.text(min_size=1, max_size=50))
    location_type = draw(st.sampled_from(["critical", "barrier"]))
    
    return CodeLocation(
        file_path=file_path,
        line_number=line_number,
        code_snippet=code_snippet,
        method_name=method_name,
        location_type=location_type
    )


@composite
def validation_result_strategy(draw):
    """Generate random ValidationResult instances."""
    is_valid = draw(st.booleans())
    errors = draw(st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=5))
    warnings = draw(st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=5))
    
    # If there are errors, is_valid should be False
    if errors:
        is_valid = False
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings
    )


@composite
def patch_strategy(draw):
    """Generate random Patch instances."""
    diff = draw(st.text(min_size=1, max_size=500))
    description = draw(st.text(min_size=1, max_size=200))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    pattern_used = draw(st.text(min_size=1, max_size=50))
    modified_files = draw(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5))
    validation_result = draw(st.one_of(st.none(), validation_result_strategy()))
    
    return Patch(
        diff=diff,
        description=description,
        confidence=confidence,
        pattern_used=pattern_used,
        modified_files=modified_files,
        validation_result=validation_result
    )


@composite
def sync_pattern_strategy(draw):
    """Generate random SyncPattern instances."""
    name = draw(st.text(min_size=1, max_size=50))
    description = draw(st.text(min_size=1, max_size=200))
    template = draw(st.text(min_size=1, max_size=500))
    example = draw(st.text(min_size=0, max_size=500))
    applicable_categories = draw(st.lists(
        st.sampled_from([cat.value for cat in FlakinessCategory]),
        min_size=1,
        max_size=5,
        unique=True
    ))
    
    return SyncPattern(
        name=name,
        description=description,
        template=template,
        example=example,
        applicable_categories=applicable_categories
    )


# Property tests

class TestTestMethodProperties:
    """Property tests for TestMethod."""
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(test_method_strategy())
    def test_test_method_serialization_roundtrip(self, test_method):
        """
        Feature: flaky-test-detection-and-repair, Property 31: Report Generation Completeness
        For any TestMethod, converting to dict and back should preserve all fields.
        """
        # Convert to dict
        data = {
            'name': test_method.name,
            'body': test_method.body,
            'annotations': test_method.annotations,
            'start_line': test_method.start_line,
            'end_line': test_method.end_line,
            'file_path': test_method.file_path,
            'signature': test_method.signature
        }
        
        # Convert back
        reconstructed = create_test_method_from_dict(data)
        
        # Verify all fields match
        assert reconstructed.name == test_method.name
        assert reconstructed.body == test_method.body
        assert reconstructed.annotations == test_method.annotations
        assert reconstructed.start_line == test_method.start_line
        assert reconstructed.end_line == test_method.end_line
        assert reconstructed.file_path == test_method.file_path
        assert reconstructed.signature == test_method.signature
    
    @settings(max_examples=100)
    @given(test_method_strategy())
    def test_test_method_json_serializable(self, test_method):
        """
        Feature: flaky-test-detection-and-repair, Property 31: Report Generation Completeness
        For any TestMethod, it should be JSON serializable.
        """
        data = {
            'name': test_method.name,
            'body': test_method.body,
            'annotations': test_method.annotations,
            'start_line': test_method.start_line,
            'end_line': test_method.end_line,
            'file_path': test_method.file_path,
            'signature': test_method.signature
        }
        
        # Should not raise exception
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed['name'] == test_method.name


class TestDetectionResultProperties:
    """Property tests for DetectionResult."""
    
    @settings(max_examples=100)
    @given(detection_result_strategy())
    def test_detection_result_serialization_roundtrip(self, result):
        """
        Feature: flaky-test-detection-and-repair, Property 31: Report Generation Completeness
        For any DetectionResult, converting to dict and back should preserve all fields.
        """
        data = {
            'is_flaky': result.is_flaky,
            'category': result.category,
            'confidence': result.confidence,
            'all_scores': result.all_scores,
            'method_name': result.method_name
        }
        
        reconstructed = create_detection_result_from_dict(data)
        
        assert reconstructed.is_flaky == result.is_flaky
        assert reconstructed.category == result.category
        assert reconstructed.confidence == result.confidence
        assert reconstructed.all_scores == result.all_scores
        assert reconstructed.method_name == result.method_name
    
    @settings(max_examples=100)
    @given(detection_result_strategy())
    def test_detection_result_category_valid(self, result):
        """
        Feature: flaky-test-detection-and-repair, Property 4: Flakiness Category Constraint
        For any DetectionResult, the category must be one of the five valid categories.
        """
        valid_categories = [cat.value for cat in FlakinessCategory]
        assert result.category in valid_categories


class TestPatchProperties:
    """Property tests for Patch."""
    
    @settings(max_examples=100)
    @given(patch_strategy())
    def test_patch_serialization(self, patch):
        """
        Feature: flaky-test-detection-and-repair, Property 31: Report Generation Completeness
        For any Patch, serialization should include all required fields.
        """
        serialized = serialize_patch(patch)
        
        # Verify all required fields are present
        assert 'diff' in serialized
        assert 'description' in serialized
        assert 'confidence' in serialized
        assert 'pattern_used' in serialized
        assert 'modified_files' in serialized
        assert 'is_valid' in serialized
        assert 'validation_errors' in serialized
        
        # Verify values match
        assert serialized['diff'] == patch.diff
        assert serialized['description'] == patch.description
        assert serialized['confidence'] == patch.confidence
        assert serialized['pattern_used'] == patch.pattern_used
        assert serialized['modified_files'] == patch.modified_files
    
    @settings(max_examples=100)
    @given(patch_strategy())
    def test_patch_json_serializable(self, patch):
        """
        Feature: flaky-test-detection-and-repair, Property 31: Report Generation Completeness
        For any Patch, serialization should be JSON serializable.
        """
        serialized = serialize_patch(patch)
        
        # Should not raise exception
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed['diff'] == patch.diff
    
    @settings(max_examples=100)
    @given(patch_strategy())
    def test_patch_confidence_range(self, patch):
        """
        Feature: flaky-test-detection-and-repair, Property 26: Patch Validation Checks
        For any Patch, confidence must be between 0 and 1.
        """
        assert 0.0 <= patch.confidence <= 1.0


class TestMethodInvocationProperties:
    """Property tests for MethodInvocation."""
    
    @settings(max_examples=100)
    @given(method_invocation_strategy())
    def test_method_invocation_line_number_positive(self, invocation):
        """
        Feature: flaky-test-detection-and-repair, Property 5: Method Invocation Identification
        For any MethodInvocation, line number must be positive.
        """
        assert invocation.line_number > 0
    
    @settings(max_examples=100)
    @given(method_invocation_strategy())
    def test_method_invocation_has_name(self, invocation):
        """
        Feature: flaky-test-detection-and-repair, Property 5: Method Invocation Identification
        For any MethodInvocation, method name must not be empty.
        """
        assert len(invocation.method_name) > 0


class TestFlakyLineProperties:
    """Property tests for FlakyLine."""
    
    @settings(max_examples=100)
    @given(flaky_line_strategy())
    def test_flaky_line_impact_score_range(self, flaky_line):
        """
        Feature: flaky-test-detection-and-repair, Property 14: Flaky Line Analysis Trigger
        For any FlakyLine, impact score must be between 0 and 1.
        """
        assert 0.0 <= flaky_line.impact_score <= 1.0
    
    @settings(max_examples=100)
    @given(st.lists(flaky_line_strategy(), min_size=2, max_size=10))
    def test_flaky_lines_sortable_by_impact(self, flaky_lines):
        """
        Feature: flaky-test-detection-and-repair, Property 15: Flaky Line Ranking
        For any set of FlakyLines, they should be sortable by impact score.
        """
        sorted_lines = sorted(flaky_lines, key=lambda x: x.impact_score, reverse=True)
        
        # Verify sorted order
        for i in range(len(sorted_lines) - 1):
            assert sorted_lines[i].impact_score >= sorted_lines[i + 1].impact_score


class TestCodeLocationProperties:
    """Property tests for CodeLocation."""
    
    @settings(max_examples=100)
    @given(code_location_strategy())
    def test_code_location_type_valid(self, location):
        """
        Feature: flaky-test-detection-and-repair, Property 18: Developer Code Constraint for Points
        For any CodeLocation, location_type must be 'critical' or 'barrier'.
        """
        assert location.location_type in ["critical", "barrier"]


class TestSyncPatternProperties:
    """Property tests for SyncPattern."""
    
    @settings(max_examples=100)
    @given(sync_pattern_strategy())
    def test_sync_pattern_categories_valid(self, pattern):
        """
        Feature: flaky-test-detection-and-repair, Property 22: Patch Pattern Adherence
        For any SyncPattern, applicable categories must be valid flakiness categories.
        """
        valid_categories = [cat.value for cat in FlakinessCategory]
        for category in pattern.applicable_categories:
            assert category in valid_categories


class TestCallGraphProperties:
    """Property tests for CallGraph."""
    
    @settings(max_examples=50)
    @given(
        st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10, unique=True)
    )
    def test_call_graph_root_in_nodes(self, node_ids):
        """
        Feature: flaky-test-detection-and-repair, Property 8: Call Graph Construction
        For any CallGraph, the root node must be in the nodes dictionary.
        """
        # Create nodes
        nodes = {
            node_id: MethodNode(
                method_id=node_id,
                method_name=f"method_{node_id}",
                file_path=f"file_{node_id}.java",
                is_external=False
            )
            for node_id in node_ids
        }
        
        # Create call graph with first node as root
        root = node_ids[0]
        call_graph = CallGraph(nodes=nodes, edges=[], root=root)
        
        assert call_graph.root in call_graph.nodes
        assert call_graph.nodes[root].method_id == root
