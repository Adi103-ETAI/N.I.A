from interface import chat


def test_listen_result_to_text_handles_string():
    assert chat._listen_result_to_text("hello world") == "hello world"


def test_listen_result_to_text_handles_dict_text():
    assert chat._listen_result_to_text({'text': 'hello'}) == 'hello'


def test_listen_result_to_text_handles_dict_output():
    assert chat._listen_result_to_text({'output': 'hey'}) == 'hey'


def test_listen_result_to_text_handles_legacy_success():
    assert chat._listen_result_to_text({'success': True, 'output': 'legacy'}) == 'legacy'


def test_listen_result_to_text_handles_common_keys():
    assert chat._listen_result_to_text({'transcript': 'transcribed'}) == 'transcribed'


def test_listen_result_to_text_handles_false_ok():
    assert chat._listen_result_to_text({'ok': False, 'error': 'no audio'}) is None


def test_listen_result_to_text_handles_none():
    assert chat._listen_result_to_text(None) is None
