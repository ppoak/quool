"""Tests for quool.util module."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Helper class for generate_usage tests (must be module-level for qualname to match)
class _GreeterForTest:
    """Greeter class for testing."""

    def greet(self):
        """Instance greet."""
        pass

    @classmethod
    def hello(cls):
        """Class hello."""
        pass

    @staticmethod
    def hi():
        """Static hi."""
        pass


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_returns_logger_instance(self):
        """Should return a logging.Logger instance."""
        logger = pytest.importorskip("quool").util.setup_logger("test_basic")
        assert isinstance(logger, logging.Logger)

    def test_default_level_is_info(self):
        """Default level should be INFO."""
        logger = pytest.importorskip("quool").util.setup_logger("test_level", replace=True)
        assert logger.level == logging.INFO

    def test_custom_level(self):
        """Should respect custom level parameter."""
        logger = pytest.importorskip("quool").util.setup_logger("test_custom_level", level=logging.DEBUG, replace=True)
        assert logger.level == logging.DEBUG

    def test_replace_creates_new_logger(self):
        """With replace=True, should create a fresh Logger even if name exists."""
        util = pytest.importorskip("quool").util
        name = "test_replace_fresh"
        logging.getLogger(name).handlers.clear()
        logger1 = util.setup_logger(name, replace=True)
        # Create a new Logger with same name - replace=True means instantiating fresh
        logger2 = util.setup_logger(name, replace=True)
        # With replace=True, it calls logging.Logger(name, level) directly,
        # which creates a new Logger instance even for the same name
        assert logger1 is not logger2

    def test_no_duplicate_handlers(self):
        """Existing logger with handlers should not get duplicate handlers."""
        util = pytest.importorskip("quool").util
        name = "test_no_dup"
        logging.getLogger(name).handlers.clear()
        logger1 = util.setup_logger(name)
        logger1.handlers.clear()
        logger2 = util.setup_logger(name)
        assert logger1 is logger2
        # Should not add handlers again
        initial_count = len(logger1.handlers)
        util.setup_logger(name)
        assert len(logger1.handlers) == initial_count

    def test_stream_handler_added_by_default(self):
        """Stream handler should be added when stream=True (default)."""
        util = pytest.importorskip("quool").util
        name = "test_stream"
        logging.getLogger(name).handlers.clear()
        logger = util.setup_logger(name, replace=True)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_no_stream_handler(self):
        """No stream handler when stream=False."""
        util = pytest.importorskip("quool").util
        name = "test_no_stream"
        logging.getLogger(name).handlers.clear()
        logger = util.setup_logger(name, stream=False, replace=True)
        assert not any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_file_handler_added(self):
        """File handler should be added when file path is provided."""
        util = pytest.importorskip("quool").util
        name = "test_file"
        logging.getLogger(name).handlers.clear()
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            logger = util.setup_logger(name, file=path, replace=True)
            assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        finally:
            for h in logger.handlers:
                h.close()
            Path(path).unlink(missing_ok=True)

    def test_size_rotation_handler(self):
        """Should use RotatingFileHandler when rotation='size'."""
        util = pytest.importorskip("quool").util
        name = "test_size_rot"
        logging.getLogger(name).handlers.clear()
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            logger = util.setup_logger(name, file=path, rotation="size", replace=True)
            from logging.handlers import RotatingFileHandler
            assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        finally:
            for h in logger.handlers:
                h.close()
            Path(path).unlink(missing_ok=True)

    def test_time_rotation_handler(self):
        """Should use TimedRotatingFileHandler when rotation='time'."""
        util = pytest.importorskip("quool").util
        name = "test_time_rot"
        logging.getLogger(name).handlers.clear()
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            logger = util.setup_logger(name, file=path, rotation="time", replace=True)
            from logging.handlers import TimedRotatingFileHandler
            assert any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers)
        finally:
            for h in logger.handlers:
                h.close()
            Path(path).unlink(missing_ok=True)

    @pytest.mark.parametrize("style", [1, 2, 3, 4])
    def test_formatter_styles(self, style):
        """Each formatter style 1-4 should produce a valid Formatter."""
        util = pytest.importorskip("quool").util
        name = f"test_style_{style}"
        logging.getLogger(name).handlers.clear()
        logger = util.setup_logger(name, stream=True, replace=True, style=style)
        assert len(logger.handlers) > 0
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                assert isinstance(h.formatter, logging.Formatter)

    def test_formatter_from_string(self):
        """Should accept a format string as style parameter."""
        util = pytest.importorskip("quool").util
        name = "test_str_style"
        logging.getLogger(name).handlers.clear()
        fmt_str = "%(levelname)s - %(message)s"
        logger = util.setup_logger(name, style=fmt_str, replace=True)
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                assert h.formatter._fmt == fmt_str

    def test_formatter_from_formatter_instance(self):
        """Should accept a logging.Formatter instance as style parameter."""
        util = pytest.importorskip("quool").util
        name = "test_fmt_instance"
        logging.getLogger(name).handlers.clear()
        formatter = logging.Formatter("%(message)s")
        logger = util.setup_logger(name, style=formatter, replace=True)
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                assert h.formatter is formatter

    def test_clear_file(self):
        """With clear=True, file should be truncated before use."""
        util = pytest.importorskip("quool").util
        name = "test_clear"
        logging.getLogger(name).handlers.clear()
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
            f.write(b"old content")
        try:
            logger = util.setup_logger(name, file=path, clear=True, replace=True)
            assert Path(path).read_text() == ""
        finally:
            for h in logger.handlers:
                h.close()
            Path(path).unlink(missing_ok=True)


class TestProxyRequest:
    """Tests for proxy_request function."""

    def test_successful_request_no_proxy(self):
        """Should return response when direct request succeeds."""
        util = pytest.importorskip("quool").util
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("quool.util.requests.request", return_value=mock_resp) as mock_req:
            resp = util.proxy_request("https://example.com")
            mock_req.assert_called_once()
            assert resp is mock_resp

    def test_successful_request_with_single_proxy(self):
        """Should try the proxy and return on success."""
        util = pytest.importorskip("quool").util
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        proxy = {"http": "http://proxy.example.com:8080"}

        with patch("quool.util.requests.request", return_value=mock_resp) as mock_req:
            resp = util.proxy_request("https://example.com", proxies=proxy)
            mock_req.assert_called_with(method="GET", url="https://example.com", proxies=proxy)
            assert resp is mock_resp

    def test_successful_request_with_proxy_list(self):
        """Should try each proxy in order and return on first success."""
        util = pytest.importorskip("quool").util
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        proxies = [{"http": "http://proxy1.com:8080"}, {"http": "http://proxy2.com:8080"}]

        with patch("quool.util.requests.request", return_value=mock_resp) as mock_req:
            resp = util.proxy_request("https://example.com", proxies=proxies)
            # First call uses first proxy
            mock_req.assert_called_with(method="GET", url="https://example.com", proxies=proxies[0])
            assert resp is mock_resp

    def test_proxy_fallback_to_direct(self):
        """Should fall back to direct request when all proxies fail."""
        util = pytest.importorskip("quool").util
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        proxies = [{"http": "http://fail.proxy.com:8080"}]

        with patch("quool.util.requests.request", side_effect=[req.exceptions.RequestException("proxy failed"), mock_resp]) as mock_req:
            resp = util.proxy_request("https://example.com", proxies=proxies, delay=0)
            assert mock_req.call_count == 2
            # Second call should be direct (no proxy dict, or proxies=None)
            second_call_kwargs = mock_req.call_args_list[1].kwargs
            # Second call should not have the failed proxy
            assert second_call_kwargs.get("proxies") != proxies[0]
            assert resp is mock_resp

    def test_post_method(self):
        """Should use POST when method='POST'."""
        util = pytest.importorskip("quool").util
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("quool.util.requests.request", return_value=mock_resp) as mock_req:
            util.proxy_request("https://example.com", method="POST", data={"key": "value"})
            mock_req.assert_called_once()
            assert mock_req.call_args[1]["method"] == "POST"
            assert mock_req.call_args[1]["data"] == {"key": "value"}

    def test_raises_on_final_failure(self):
        """Should raise RequestException when final direct request also fails."""
        util = pytest.importorskip("quool").util
        import requests as req

        with patch("quool.util.requests.request", side_effect=req.exceptions.RequestException("failed")):
            with pytest.raises(req.exceptions.RequestException):
                util.proxy_request("https://example.com")

    def test_proxy_list_from_iterable(self):
        """Should accept tuple as proxies input."""
        util = pytest.importorskip("quool").util
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        proxies = ({"http": "http://proxy1.com:8080"}, {"http": "http://proxy2.com:8080"})

        with patch("quool.util.requests.request", return_value=mock_resp) as mock_req:
            util.proxy_request("https://example.com", proxies=proxies)
            assert mock_req.call_count == 1


class TestNotifyTask:
    """Tests for notify_task decorator."""

    def test_decorator_returns_callable(self):
        """Decorator should return a callable wrapper."""
        util = pytest.importorskip("quool").util
        @util.notify_task(sender="a@a.com", password="pass", receiver="b@b.com", smtp_server="smtp.example.com", smtp_port=587)
        def dummy():
            return 42
        assert callable(dummy)

    def test_decorator_preserves_function_metadata(self):
        """Decorator should preserve original function name and docstring."""
        util = pytest.importorskip("quool").util
        @util.notify_task(sender="a@a.com", password="pass", receiver="b@b.com", smtp_server="smtp.example.com", smtp_port=587)
        def my_func():
            """Original docstring."""
            return 1
        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "Original docstring."

    def test_successful_execution_returns_result(self):
        """On success, should return the original function's result."""
        util = pytest.importorskip("quool").util

        with patch("quool.util.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            @util.notify_task(sender="a@a.com", password="pass", receiver="b@b.com", smtp_server="smtp.example.com", smtp_port=587)
            def add(x, y):
                return x + y

            result = add(1, 2)
            assert result == 3
            # Should have sent an email
            mock_server.starttls.assert_called_once()
            mock_server.sendmail.assert_called_once()

    def test_exception_caught_and_returned_as_string(self):
        """On exception, should catch it, send failure email, and return str of exception."""
        util = pytest.importorskip("quool").util

        with patch("quool.util.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            @util.notify_task(sender="a@a.com", password="pass", receiver="b@b.com", smtp_server="smtp.example.com", smtp_port=587)
            def bad_func():
                raise ValueError("test error")

            result = bad_func()
            assert result == "test error"
            # Should still send email on failure
            assert mock_server.sendmail.called

    def test_string_result_handling(self):
        """String results should be included directly in message body."""
        util = pytest.importorskip("quool").util

        with patch("quool.util.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            @util.notify_task(sender="a@a.com", password="pass", receiver="b@b.com", smtp_server="smtp.example.com", smtp_port=587)
            def return_str():
                return "simple string result"

            result = return_str()
            assert result == "simple string result"

    def test_dataframe_result_handling(self):
        """DataFrame results should be converted to markdown."""
        pd = pytest.importorskip("pandas")
        util = pytest.importorskip("quool").util

        with patch("quool.util.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            @util.notify_task(sender="a@a.com", password="pass", receiver="b@b.com", smtp_server="smtp.example.com", smtp_port=587)
            def return_df():
                return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

            result = return_df()
            assert isinstance(result, pd.DataFrame)


class TestGenerateUsage:
    """Tests for generate_usage function."""

    def test_returns_string(self):
        """Should return a markdown-formatted string."""
        util = pytest.importorskip("quool").util

        def dummy_func(x, y):
            """Add two numbers."""
            return x + y

        result = util.generate_usage(dummy_func)
        assert isinstance(result, str)

    def test_contains_function_name(self):
        """Output should contain the function name."""
        util = pytest.importorskip("quool").util

        def my_calculator(a, b):
            """Calculate something."""
            return a * b

        result = util.generate_usage(my_calculator)
        assert "my_calculator" in result

    def test_contains_docstring(self):
        """Output should contain the docstring summary."""
        util = pytest.importorskip("quool").util

        def adder(x, y):
            """Adds two values together."""
            return x + y

        result = util.generate_usage(adder)
        assert "Adds two values" in result

    def test_class_returns_markdown(self):
        """Should generate documentation for a class."""
        util = pytest.importorskip("quool").util

        class MyClass:
            """A sample class for testing."""

            def __init__(self, name):
                self.name = name

            def greet(self):
                """Return a greeting."""
                return f"Hello, {self.name}"

        result = util.generate_usage(MyClass)
        assert isinstance(result, str)
        assert "MyClass" in result

    def test_output_to_file(self):
        """Should write output to file when output_path is specified."""
        util = pytest.importorskip("quool").util

        def hello(name):
            """Say hello."""
            return f"Hello, {name}"

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="r") as f:
            path = f.name

        try:
            util.generate_usage(hello, output_path=path)
            content = Path(path).read_text(encoding="utf-8")
            assert "hello" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_include_parameters_section(self):
        """Should include parameters when include_sections includes 'parameters'."""
        util = pytest.importorskip("quool").util

        def greet(name, greeting="Hello", times=1):
            """Greet someone."""
            pass

        result = util.generate_usage(greet, include_sections=["parameters"])
        assert "name" in result

    def test_raises_on_non_class_non_callable(self):
        """Should raise TypeError for non-class, non-callable input."""
        util = pytest.importorskip("quool").util

        with pytest.raises(TypeError):
            util.generate_usage(42)

    def test_render_tables_false(self):
        """Should support render_tables=False."""
        util = pytest.importorskip("quool").util

        def add(x, y):
            """Add."""
            return x + y

        result = util.generate_usage(add, render_tables=False)
        assert isinstance(result, str)
        assert "add" in result

    def test_heading_level(self):
        """Should respect heading_level parameter."""
        util = pytest.importorskip("quool").util

        def hello():
            """Hello."""
            pass

        result = util.generate_usage(hello, heading_level=3)
        # h3 headings should appear as ###
        assert "###" in result

    def test_include_signature(self):
        """Should include signature when include_signature=True."""
        util = pytest.importorskip("quool").util

        def multiply(a, b):
            """Multiply."""
            return a * b

        result = util.generate_usage(multiply, include_signature=True)
        assert "multiply" in result

    def test_method_kinds_filter(self):
        """Should respect method_kinds parameter when filtering methods."""
        util = pytest.importorskip("quool").util

        # Without filtering - all methods present
        all_result = util.generate_usage(_GreeterForTest)
        assert "hello" in all_result
        assert "greet" in all_result
        assert "hi" in all_result

        # With method_kinds=['class'] - only classmethod present
        classmethod_only = util.generate_usage(_GreeterForTest, method_kinds=["class"])
        assert "hello" in classmethod_only
        assert "greet" not in classmethod_only
        assert "hi" not in classmethod_only
