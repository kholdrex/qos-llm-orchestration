from qos_orchestration.scoring import code_program, new_marker, score_long, score_math, score_qa


def test_qa_normalised_containment():
    assert score_qa("It was the Golan Heights.", {"answers": ["golan heights"]}) == 1.0
    assert score_qa("No idea", {"answers": ["Jordan"]}) == 0.0


def test_math_prefers_explicit_answer_line():
    assert score_math("3 apples and 4 pears\nAnswer: 1,200", {"answer": "1200"}) == 1.0
    assert score_math("Answer: 7", {"answer": "8"}) == 0.0


def test_code_program_keeps_prompt_helpers():
    ref = {
        "prompt": 'def helper():\n    return 1\n\ndef f(x):\n    """doc"""\n',
        "entry_point": "f",
        "test": "def check(c):\n    assert c(1) == 2",
    }
    marker = new_marker()
    program = code_program("```python\ndef f(x):\n    return x + helper()\n```", ref, marker)
    assert program.rstrip().endswith(f"print({marker!r})")
    assert marker != new_marker()
    exec(program, {})


def test_long_form_instruction_share():
    ref = {
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"relation": "at least", "num_words": 5}],
    }
    assert score_long("one two three four five six", ref)[0] == 1.0
    assert score_long("too short", ref)[0] == 0.0


def test_language_checks_are_deterministic():
    ref = {"instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}
    assert len({score_long("hello world", ref)[0] for _ in range(30)}) == 1
