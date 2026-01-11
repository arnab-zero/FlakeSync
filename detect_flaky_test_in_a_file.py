# print("\n📝 Example Detection:")
# result = detect_flaky_test(example_code)
# print(f"Result: {result}")


import os
import re
import json
from pprint import pprint


try:
	from flaky_test_detector_main import detect_flaky_test
except Exception as e:
	detect_flaky_test = None
	import sys
	print("Warning: could not import detect_flaky_test from flaky_test_detector_main:", e, file=sys.stderr)


def extract_test_methods(java_source: str) -> list[str]: 
	"""Extract test methods annotated with @Test from a Java source string.

	Returns a list of method source strings (including the @Test annotation and method body).
	Uses a brace-matching approach so method bodies containing nested braces are handled.
	"""
	tests = []
	for m in re.finditer(r"@Test", java_source):
		start = m.start()
		# Find first brace after the annotation which should be the method body start
		brace_idx = java_source.find('{', start)
		if brace_idx == -1:
			continue

		# Walk forward to find matching closing brace
		idx = brace_idx + 1
		depth = 1
		length = len(java_source)
		while idx < length and depth > 0:
			ch = java_source[idx]
			if ch == '{':
				depth += 1
			elif ch == '}':
				depth -= 1
			idx += 1

		if depth == 0:
			method_src = java_source[start:idx]
			tests.append(method_src.strip())

	return tests


def main():
	repo_dir = os.path.dirname(__file__)
	java_path = os.path.join(repo_dir, 'test_file.java')
	if not os.path.exists(java_path):
		print(f"test file not found at {java_path}")
		return

	with open(java_path, 'r', encoding='utf-8') as f:
		src = f.read()

	methods = extract_test_methods(src)
	if not methods:
		print("No @Test methods found in test_file.java")
		return

	print(f"Found {len(methods)} test method(s). Running detection on each...")

	for i, method_src in enumerate(methods, start=1):
		header = f"--- Test #{i} ---"
		print(header)
		# For readability, show the method signature line (first 2 lines)
		preview = '\n'.join(method_src.splitlines()[:4])
		print(preview)

		if detect_flaky_test is None:
			print("detect_flaky_test() not available (import failed). Skipping analysis.")
			continue

		try:
			result = detect_flaky_test(method_src)
		except Exception as e:
			result = {"error": str(e)}

		# Pretty-print JSON-like output
		try:
			print(json.dumps(result, indent=2, ensure_ascii=False))
		except TypeError:
			pprint(result)


if __name__ == '__main__':
	main()
