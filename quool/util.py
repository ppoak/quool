import inspect
import textwrap
import os
import re
import random
import time
import traceback
import mimetypes
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import (
    get_origin,
    get_args,
    List,
    Optional,
    Union,
    Literal,
)

import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import markdown
import pandas as pd
from retry import retry


def setup_logger(
    name,
    level: str = logging.INFO,
    replace: bool = False,
    stream: bool = True,
    file: Union[Path, str] = None,
    clear: bool = False,
    style: Union[int, str] = 1,
    rotation: str = None,
    max_bytes: int = None,
    backup_count: int = None,
    when: str = None,
    interval: int = None,
) -> logging.Logger:
    """Create and configure a logger with optional stream/file handlers and rotation.

    This helper creates (or returns) a logging.Logger configured with optional
    console (stream) and file handlers. File handlers may use size-based or
    time-based rotation. If a logger with the same name already has handlers,
    the existing logger is returned unchanged (to avoid duplicate handlers).

    Args:
        name (str): Name of the logger.
        level (int or str, optional): Logging level (e.g. logging.INFO or 'INFO').
            Default: logging.INFO.
        replace (bool, optional): If True, instantiate a new Logger object even
            if one exists. If False, use logging.getLogger(name). Default: False.
        stream (bool, optional): If True, add a StreamHandler to emit logs to
            stderr/stdout. Default: True.
        file (Path or str, optional): Path to a file to also write logs to. If
            provided, a file handler is attached (regular or rotating depending
            on `rotation`). Default: None.
        clear (bool, optional): If True and `file` is provided, truncate the
            file before use. Default: False.
        style (int or logging.Formatter or str, optional): Select a built-in
            formatter style by integer (1..4). If not an int matching a built-in
            style, the value is used directly (e.g. a logging.Formatter instance
            or a custom format string). Default: 1.
        rotation (str, optional): Rotation mode for the file handler. Supported
            values: "size" (RotatingFileHandler), "time" (TimedRotatingFileHandler),
            or None (no rotation). Default: None.
        max_bytes (int, optional): Max bytes for size-based rotation. If not
            provided, defaults to 10 * 1024 * 1024 (10 MB).
        backup_count (int, optional): Number of backup files to keep for rotation.
            Defaults: 5 for size-based rotation, 7 for time-based rotation when not set.
        when (str, optional): When parameter for time-based rotation
            (e.g. 'midnight'). Default: "midnight" when rotation == "time".
        interval (int, optional): Interval for time-based rotation (in units
            defined by `when`). Default: 1.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if file and clear:
        Path(file).write_text("")

    if not replace:
        logger = logging.getLogger(name)
    else:
        logger = logging.Logger(name, level=level)

    if logger.hasHandlers():
        return logger  # Avoid adding handlers multiple times

    logger.setLevel(level)

    # Define formatter styles (available to both stream and file handlers)
    formatter_styles = {
        1: logging.Formatter("[%(levelname)s]@[%(asctime)s]-[%(name)s]: %(message)s"),
        2: logging.Formatter(
            "[%(levelname)s]@[%(asctime)s]-[%(name)s]@[%(module)s:%(lineno)d]: %(message)s"
        ),
        3: logging.Formatter(
            "[%(levelname)s]@[%(asctime)s]-[%(name)s]@[%(module)s:%(lineno)d#%(funcName)s]: %(message)s"
        ),
        4: logging.Formatter(
            "[%(levelname)s]@[%(asctime)s]-[%(name)s]@[%(module)s:%(lineno)d#%(funcName)s~%(process)d:%(threadName)s]: %(message)s"
        ),
    }

    # Resolve formatter from style parameter
    formatter = None
    if isinstance(style, int):
        formatter = formatter_styles.get(style, formatter_styles[1])
    else:
        # style may be a logging.Formatter instance or a format string
        formatter = style

    if isinstance(formatter, str):
        formatter = logging.Formatter(formatter)

    if not isinstance(formatter, logging.Formatter):
        # Fallback to default if user provided an unexpected type
        formatter = formatter_styles[1]

    # Add stream handler if requested
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add file handler if specified
    if file:
        if rotation == "size":
            file_handler = RotatingFileHandler(
                file,
                encoding="utf-8",
                maxBytes=max_bytes or 10 * 1024 * 1024,
                backupCount=backup_count or 5,
            )
        elif rotation == "time":
            file_handler = TimedRotatingFileHandler(
                file,
                encoding="utf-8",
                when=when or "midnight",
                interval=interval or 1,
                backupCount=backup_count or 7,
            )
        else:
            file_handler = logging.FileHandler(file, encoding="utf-8")

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def notify_task(
    sender: str = None,
    password: str = None,
    receiver: str = None,
    smtp_server: str = None,
    smtp_port: int = None,
    cc: str = None,
):
    """Decorator that runs a task and sends an email notification with its result.

    This decorator executes the wrapped function and sends an email containing the
    function result, execution parameters, start/end times and duration. Common return
    types receive special formatting:
      - pandas.DataFrame / pandas.Series: converted to markdown (head/tail if large).
      - dict: converted to a DataFrame then to markdown.
      - str or other objects: converted to str().
    If the wrapped function raises an exception, the decorator captures the traceback,
    sends a failure email containing the formatted traceback, and returns the exception's
    string representation (it does not re-raise the original exception).

    The decorator also parses markdown image/file links in the message:
      - Image files (.png, .jpg, .jpeg, .gif) are embedded inline using Content-ID (CID).
      - Text files are attached as text/plain attachments.
      - Non-text files are attached as binary (octet-stream) with base64 encoding.

    SMTP credentials and recipients can be provided as parameters or via environment
    variables when parameters are None:
      NOTIFY_TASK_SENDER, NOTIFY_TASK_PASSWORD, NOTIFY_TASK_RECEIVER,
      NOTIFY_TASK_SMTP_SERVER, NOTIFY_TASK_SMTP_PORT, NOTIFY_TASK_CC

    Note: The current implementation contains a probable bug where smtp_port is assigned
    from smtp_server instead of the intended environment variable. Verify smtp_port
    before use.

    Args:
        sender (str, optional): Sender email address. If None, read from
            NOTIFY_TASK_SENDER.
        password (str, optional): Sender email password or app-specific password. If None,
            read from NOTIFY_TASK_PASSWORD.
        receiver (str, optional): Comma-separated recipient addresses. If None, read
            from NOTIFY_TASK_RECEIVER.
        smtp_server (str, optional): SMTP server host. If None, read from
            NOTIFY_TASK_SMTP_SERVER.
        smtp_port (int, optional): SMTP server port. If None, read from
            NOTIFY_TASK_SMTP_PORT.
        cc (str, optional): Comma-separated CC addresses. If None, read from
            NOTIFY_TASK_CC.

    Returns:
        Callable: A decorator that wraps the target function. The wrapped function will:
            - Execute the original function and return its result on success.
            - On exception, catch the exception, send a failure notification, and return
              the exception's string representation.

    Raises:
        smtplib.SMTPException: If SMTP connection, authentication, or sending fails.
        OSError/FileNotFoundError: If referenced local files in the markdown cannot be
            read when attaching or embedding.
        UnicodeDecodeError: While attaching a file as text if decoding fails (the code
            falls back to binary attachment for such cases, but file I/O may still raise).

    Example:
        @notify_task()
        def my_job(x, y):
            return x + y

        # Calling my_job(1, 2) will send an email titled like:
        # "Task my_job success" and include the result, parameters, and duration.
    """

    sender = sender or os.getenv("NOTIFY_TASK_SENDER")
    password = password or os.getenv("NOTIFY_TASK_PASSWORD")
    receiver = receiver or os.getenv("NOTIFY_TASK_RECEIVER")
    smtp_server = smtp_server or os.getenv("NOTIFY_TASK_SMTP_SERVER")
    smtp_port = smtp_port or os.getenv("NOTIFY_TASK_SMTP_PORT")
    cc = cc or os.getenv("NOTIFY_TASK_CC")

    def wrapper(task):
        @wraps(task)
        def wrapper(*args, **kwargs):
            try:
                success = True
                begin = pd.to_datetime("now")
                result = task(*args, **kwargs)
                end = pd.to_datetime("now")
                duration = end - begin
                if isinstance(result, str):
                    result_str = result
                elif isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                    if len(result) > 10:
                        result_str = (
                            result.head().to_markdown()
                            + "\n\n...\n\n"
                            + result.tail().to_markdown()
                        )
                    else:
                        result_str = result.to_markdown()
                elif isinstance(result, dict):
                    result_str = pd.DataFrame(result).to_markdown()
                else:
                    result_str = str(result)
                args = [
                    str(arg).replace(">", "&gt;").replace("<", "&lt;") for arg in args
                ]
                kwargs = {
                    key: str(value).replace(">", "&gt;").replace("<", "&lt;")
                    for key, value in kwargs.items()
                }
                message = (
                    f"{result_str}\n\n"
                    f"> *Parameters: {args} {kwargs}*\n\n"
                    f"> *Run from {begin} to {end} ({duration})*"
                )
            except Exception as e:
                success = False
                result = str(e)
                end = pd.to_datetime("now")
                args = [
                    str(arg).replace(">", "&gt;").replace("<", "&lt;") for arg in args
                ]
                kwargs = {
                    key: str(value).replace(">", "&gt;").replace("<", "&lt;")
                    for key, value in kwargs.items()
                }
                duration = end - begin
                message = (
                    "```\n{traces}\n```\n\n"
                    "> *Parameters: {args} {kwargs}*\n\n"
                    "> *Run from {begin} to {end} ({duration})*"
                ).format(
                    traces="\n".join(
                        [
                            trace.replace("^", "")
                            for trace in traceback.format_exception(
                                type(e), e, e.__traceback__
                            )
                        ]
                    ),
                    args=args,
                    kwargs=kwargs,
                    begin=begin,
                    end=end,
                    duration=duration,
                )
            finally:
                subject = f"Task {task.__name__} {'success' if success else 'failure'}"
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender, password)
                content = MIMEMultipart("related")
                content["From"] = sender
                content["To"] = receiver
                if cc:
                    content["Cc"] = cc
                content["Subject"] = subject
                html_body = markdown.markdown(
                    message, extensions=["tables", "fenced_code", "codehilite", "extra"]
                )
                # Find all paths in the markdown using a regular expression
                file_paths = re.findall(r"!\[.*?\]\((.*?)\)", message)

                # Attach images and files as needed
                for i, file_path in enumerate(file_paths):
                    file = Path(file_path)
                    if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
                        with file.open("rb") as bin:
                            img_data = bin.read()
                            # Create a unique content ID
                            cid = f"image{i}"
                            image_mime = MIMEImage(img_data)
                            image_mime.add_header("Content-ID", f"<{cid}>")
                            image_mime.add_header(
                                "Content-Disposition", "inline", filename=file.name
                            )
                            content.attach(image_mime)
                            # Replace the file path in the HTML body with a cid reference
                            html_body = html_body.replace(file_path, f"cid:{cid}")
                    else:
                        ctype, _ = mimetypes.guess_type(file.name)
                        if ctype and ctype.startswith("text/"):
                            part = MIMEText(file.read_text("utf-8"), "plain", "utf-8")
                            part.add_header(
                                "Content-Disposition",
                                f'attachment; filename="{file.name}"',
                            )
                            content.attach(part)
                        else:
                            ctype = ctype or "application/octet-stream"
                            main, sub = ctype.split("/", 1)
                            part = MIMEBase(main, sub)
                            part.set_payload(file.read_bytes())
                            encoders.encode_base64(part)
                            part.add_header(
                                "Content-Disposition",
                                f'attachment; filename="{file.name}"',
                            )
                            part.add_header(
                                "Content-Type", f'{ctype}; name="{file.name}"'
                            )
                            content.attach(part)

                # Update the HTML part with embedded image references
                content.attach(MIMEText(html_body, "html"))

                # Prepare the recipient list, including CC recipients
                recipient_list = receiver.split(",")
                if cc:
                    recipient_list += cc.split(",")
                server.sendmail(sender, recipient_list, content.as_string())

            return result

        return wrapper

    return wrapper


@retry(exceptions=(requests.exceptions.RequestException,), tries=5, delay=1, backoff=2)
def proxy_request(
    url: str,
    method: str = "GET",
    proxies: Union[dict, list] = None,
    delay: float = 1,
    **kwargs,
) -> requests.Response:
    """Request a URL using an optional list of proxy configurations, falling back to a direct request.

    This function will attempt to perform an HTTP request using each provided proxy in turn.
    If a proxy attempt raises a requests.exceptions.RequestException, it will wait `delay`
    seconds and try the next proxy. If all proxies fail (or if no proxies are provided),
    a direct request (no proxy) is attempted. The function raises if the final request
    fails; note that the retry decorator will retry the whole function on RequestException.

    Args:
        url (str): Target URL.
        method (str, optional): HTTP method to use (e.g., "GET", "POST"). Defaults to "GET".
        proxies (dict or list[dict] or None, optional): A single requests-style proxies dict
            (e.g. {"http": "...", "https": "..."}) or a list of such dicts. If None, no proxies
            will be tried. Defaults to None.
        delay (float, optional): Seconds to sleep between proxy attempts on failure. Defaults to 1.
        **kwargs: Additional keyword arguments forwarded to requests.request (e.g., headers, data).

    Returns:
        requests.Response: The successful requests Response object.

    Raises:
        requests.exceptions.RequestException: If the final request (after trying proxies and direct)
            fails. Note that the retry decorator may re-invoke this function on such exceptions.
    """
    # Normalize proxies into a list of dicts
    if proxies is None:
        proxy_list = []
    elif isinstance(proxies, dict):
        proxy_list = [proxies]
    elif isinstance(proxies, list):
        proxy_list = proxies.copy()
    else:
        # Accept any iterable of proxy dicts (e.g., tuple)
        proxy_list = list(proxies)

    # Use a deepcopy to avoid mutating caller data
    proxy_list = deepcopy(proxy_list)

    for proxy in proxy_list:
        try:
            response = requests.request(method=method, url=url, proxies=proxy, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            time.sleep(delay)

    # Try a direct request if proxies are exhausted or none provided
    response = requests.request(method=method, url=url, **kwargs)
    response.raise_for_status()
    return response


def _generate_class_usage(
    target: object,
    output_path: Optional[str] = None,
    *,
    # Class-scope controls
    include_private: bool = False,
    include_inherited: bool = False,
    include_properties: bool = True,
    include_methods: bool = True,
    method_kinds: tuple[str, ...] = ("instance", "class", "static"),
    method_include: Optional[list[str]] = None,
    method_exclude: Optional[list[str]] = None,
    attribute_include: Optional[list[str]] = None,
    attribute_exclude: Optional[list[str]] = None,
    # Rendering controls
    sort_methods: Literal["name", "kind", "none"] = "name",
    render_tables: bool = True,
    include_signature: bool = True,
    include_sections: Optional[
        Literal[
            "summary",
            "description",
            "attributes",
            "methods",
            "parameters",
            "returns",
            "raises",
            "examples",
        ]
    ] = None,
    heading_level: int = 2,
) -> str:
    """Internal function for generating usage documentation."""
    # Validate/normalize rendering controls (keep original behavior)
    allowed_sections = {
        "summary",
        "description",
        "attributes",
        "methods",
        "parameters",
        "returns",
        "raises",
        "examples",
    }
    sec = set(include_sections) if include_sections else allowed_sections
    sec = sec.intersection(allowed_sections)
    if sort_methods not in ("name", "kind", "none"):
        sort_methods = "name"
    base_h = max(1, int(heading_level))
    h1 = "#" * base_h
    h2 = "#" * (base_h + 1)
    h3 = "#" * (base_h + 2)

    if not inspect.isclass(target):
        raise TypeError("target must be a class")

    obj_module = getattr(target, "__module__", "")
    obj_qualname = getattr(
        target, "__qualname__", getattr(target, "__name__", str(target))
    )
    obj_name = getattr(target, "__name__", str(target))

    lines: list[str] = []

    # ----- Parse class docstring -----
    raw_doc = inspect.getdoc(target)
    cls_summary = ""
    cls_description = ""
    cls_attributes = []
    cls_examples = []
    if raw_doc:
        d = textwrap.dedent(raw_doc).strip("\n")
        L = d.splitlines()
        n = len(L)
        i = 0
        summary_lines = []
        while i < n and L[i].strip():
            summary_lines.append(L[i].strip())
            i += 1
        cls_summary = " ".join(summary_lines).strip()
        while i < n and not L[i].strip():
            i += 1
        section = None
        section_buffer = {
            "description": [],
            "Args": [],
            "Returns": [],
            "Raises": [],
            "Attributes": [],
            "Examples": [],
        }
        headers = set(section_buffer.keys())
        while i < n:
            line = L[i]
            s = line.strip()
            if s.endswith(":"):
                h = s[:-1]
                if h in headers:
                    section = h
                    i += 1
                    continue
            if section:
                section_buffer[section].append(line)
            else:
                section_buffer["description"].append(line)
            i += 1
        cls_description = "\n".join(
            [x.rstrip() for x in section_buffer["description"]]
        ).strip()

        # Attributes items
        items = []
        current = None
        for b in section_buffer["Attributes"]:
            if not b.strip():
                continue
            m = re.match(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\(([^)]*)\))?\s*:\s*(.*)$", b
            )
            if m:
                name_i = m.group(1)
                type_i = m.group(3)
                desc_i = m.group(4).strip()
                current = {
                    "name": name_i,
                    "type": (type_i.strip() if type_i else None),
                    "desc": desc_i,
                }
                items.append(current)
            else:
                if current:
                    current["desc"] += " " + b.strip()
        cls_attributes = items

        # Examples blocks
        if section_buffer["Examples"]:
            ex_blocks = []
            block = []
            for l in section_buffer["Examples"]:
                if l.strip():
                    block.append(l.rstrip())
                else:
                    if block:
                        ex_blocks.append("\n".join(block))
                        block = []
            if block:
                ex_blocks.append("\n".join(block))
            cls_examples = ex_blocks

    # ----- Collect attributes (merge @property) -----
    attr_map: dict[str, dict] = {}
    for a in cls_attributes:
        attr_map[a["name"]] = {
            "name": a["name"],
            "type": a.get("type"),
            "desc": a.get("desc", ""),
        }

    if include_properties:
        for name, _ in inspect.getmembers(target):
            if name.startswith("_") and not include_private:
                continue
            try:
                stat = inspect.getattr_static(target, name)
            except Exception:
                stat = None
            if not isinstance(stat, property):
                continue

            # property doc (summary/desc)
            p_raw = inspect.getdoc(stat.fget)
            p_summary = ""
            p_description = ""
            if p_raw:
                pd = textwrap.dedent(p_raw).strip("\n")
                PL = pd.splitlines()
                pn = len(PL)
                pi = 0
                p_sum_lines = []
                while pi < pn and PL[pi].strip():
                    p_sum_lines.append(PL[pi].strip())
                    pi += 1
                p_summary = " ".join(p_sum_lines).strip()
                while pi < pn and not PL[pi].strip():
                    pi += 1
                p_section = None
                p_buffer = {
                    "description": [],
                    "Args": [],
                    "Returns": [],
                    "Raises": [],
                    "Attributes": [],
                    "Examples": [],
                }
                p_headers = set(p_buffer.keys())
                while pi < pn:
                    pline = PL[pi]
                    ps = pline.strip()
                    if ps.endswith(":"):
                        ph = ps[:-1]
                        if ph in p_headers:
                            p_section = ph
                            pi += 1
                            continue
                    if p_section:
                        p_buffer[p_section].append(pline)
                    else:
                        p_buffer["description"].append(pline)
                    pi += 1
                p_description = "\n".join(
                    [x.rstrip() for x in p_buffer["description"]]
                ).strip()

            # property type from return annotation (inline ann->str logic; keep original)
            try:
                pann = stat.fget.__annotations__.get("return", inspect._empty)
            except Exception:
                pann = inspect._empty

            if pann is inspect._empty:
                ptype_str = "any"
            else:
                porigin = get_origin(pann)
                pargs = get_args(pann)
                if porigin is Union:
                    non_none = [a for a in pargs if a is not type(None)]
                    if len(non_none) == 1 and len(pargs) == 2 and type(None) in pargs:
                        inner = non_none[0]
                        try:
                            ptype_str = f"Optional[{inner.__name__}]"
                        except Exception:
                            ptype_str = f"Optional[{str(inner)}]"
                    else:
                        parts = []
                        for a in pargs:
                            try:
                                parts.append(a.__name__)
                            except Exception:
                                parts.append(str(a))
                        ptype_str = " or ".join(parts)
                elif porigin in (list,):
                    try:
                        inner = pargs[0] if pargs else None
                        if inner is None:
                            ptype_str = "List[any]"
                        else:
                            try:
                                ptype_str = f"List[{inner.__name__}]"
                            except Exception:
                                ptype_str = f"List[{str(inner)}]"
                    except Exception:
                        ptype_str = "List[any]"
                elif porigin in (dict,):
                    try:
                        k = pargs[0] if pargs else None
                        v = pargs[1] if len(pargs) > 1 else None
                        k_str = getattr(k, "__name__", str(k)) if k else "any"
                        v_str = getattr(v, "__name__", str(v)) if v else "any"
                        ptype_str = f"Dict[{k_str}, {v_str}]"
                    except Exception:
                        ptype_str = "Dict[any, any]"
                elif porigin in (tuple,):
                    try:
                        inner = (
                            ", ".join(getattr(a, "__name__", str(a)) for a in pargs)
                            if pargs
                            else ""
                        )
                        ptype_str = f"Tuple[{inner}]"
                    except Exception:
                        ptype_str = "Tuple"
                else:
                    try:
                        ptype_str = pann.__name__
                    except Exception:
                        ptype_str = str(pann)

            entry = {
                "name": name,
                "type": None if ptype_str == "any" else ptype_str,
                "desc": p_summary
                or p_description
                or attr_map.get(name, {}).get("desc", ""),
            }
            if name in attr_map:
                if not attr_map[name].get("type") and entry["type"]:
                    attr_map[name]["type"] = entry["type"]
                if not attr_map[name].get("desc") and entry["desc"]:
                    attr_map[name]["desc"] = entry["desc"]
            else:
                attr_map[name] = entry

    attrs = list(attr_map.values())
    if attribute_include:
        inc = set(attribute_include)
        attrs = [a for a in attrs if a["name"] in inc]
    if attribute_exclude:
        exc = set(attribute_exclude)
        attrs = [a for a in attrs if a["name"] not in exc]

    # ----- Collect methods -----
    methods = []
    if include_methods:
        for name, member in inspect.getmembers(target):
            if name.startswith("_") and not include_private:
                continue
            if not callable(member):
                continue
            try:
                stat = inspect.getattr_static(target, name)
            except Exception:
                stat = None

            if isinstance(stat, staticmethod):
                kind = "static"
                func = stat.__func__
            elif isinstance(stat, classmethod):
                kind = "class"
                func = stat.__func__
            elif (
                inspect.isfunction(stat)
                or inspect.ismethod(stat)
                or inspect.isroutine(stat)
            ):
                kind = "instance"
                func = member if inspect.isfunction(member) else getattr(target, name)
            else:
                continue

            if kind not in method_kinds:
                continue
            if not include_inherited:
                qn = getattr(func, "__qualname__", "")
                base = qn.split(".")[0] if qn else ""
                if base and base != target.__name__:
                    continue
            if method_include and name not in set(method_include):
                continue
            if method_exclude and name in set(method_exclude):
                continue

            sig = inspect.signature(func)

            # Parse method docstring
            m_raw = inspect.getdoc(func)
            m_summary = ""
            m_description = ""
            m_args = []
            m_returns = None
            m_raises = []
            m_examples = []
            if m_raw:
                md = textwrap.dedent(m_raw).strip("\n")
                ML = md.splitlines()
                mn = len(ML)
                mi = 0
                m_sum_lines = []
                while mi < mn and ML[mi].strip():
                    m_sum_lines.append(ML[mi].strip())
                    mi += 1
                m_summary = " ".join(m_sum_lines).strip()
                while mi < mn and not ML[mi].strip():
                    mi += 1
                m_section = None
                m_buffer = {
                    "description": [],
                    "Args": [],
                    "Returns": [],
                    "Raises": [],
                    "Attributes": [],
                    "Examples": [],
                }
                m_headers = set(m_buffer.keys())
                while mi < mn:
                    ml = ML[mi]
                    ms = ml.strip()
                    if ms.endswith(":"):
                        mh = ms[:-1]
                        if mh in m_headers:
                            m_section = mh
                            mi += 1
                            continue
                    if m_section:
                        m_buffer[m_section].append(ml)
                    else:
                        m_buffer["description"].append(ml)
                    mi += 1
                m_description = "\n".join(
                    [x.rstrip() for x in m_buffer["description"]]
                ).strip()

                # Args
                current = None
                for b in m_buffer["Args"]:
                    if not b.strip():
                        continue
                    mm = re.match(
                        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\(([^)]*)\))?\s*:\s*(.*)$",
                        b,
                    )
                    if mm:
                        an = mm.group(1)
                        at = mm.group(3)
                        ad = mm.group(4).strip()
                        current = {
                            "name": an,
                            "type": (at.strip() if at else None),
                            "desc": ad,
                        }
                        m_args.append(current)
                    else:
                        if current:
                            current["desc"] += " " + b.strip()

                # Returns
                ret_lines = [l for l in m_buffer["Returns"] if l.strip()]
                if ret_lines:
                    mmr = re.match(r"^\s*([^:]+?)\s*:\s*(.*)$", ret_lines[0].strip())
                    if mmr:
                        rt = mmr.group(1).strip()
                        rd = mmr.group(2).strip()
                        extra = " ".join(l.strip() for l in ret_lines[1:]).strip()
                        if extra:
                            rd = (rd + " " + extra).strip()
                        m_returns = {"type": rt, "desc": rd}
                    else:
                        rd = " ".join(l.strip() for l in ret_lines).strip()
                        m_returns = {"type": None, "desc": rd}

                # Raises
                current = None
                for b in m_buffer["Raises"]:
                    if not b.strip():
                        continue
                    rm = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(.*)$", b)
                    if rm:
                        et = rm.group(1)
                        ed = rm.group(2).strip()
                        current = {"type": et, "desc": ed}
                        m_raises.append(current)
                    else:
                        if current:
                            current["desc"] += " " + b.strip()

                # Examples
                if m_buffer["Examples"]:
                    ex_blocks = []
                    block = []
                    for l in m_buffer["Examples"]:
                        if l.strip():
                            block.append(l.rstrip())
                        else:
                            if block:
                                ex_blocks.append("\n".join(block))
                                block = []
                    if block:
                        ex_blocks.append("\n".join(block))
                    m_examples = ex_blocks

            # Build parameters with merged doc + annotations
            args_doc_map = {a["name"]: a for a in m_args}
            params = []
            for pname, p in sig.parameters.items():
                if pname == "self" and kind == "instance":
                    continue
                if pname == "cls" and kind == "class":
                    continue

                pann = p.annotation
                if pann is inspect._empty:
                    ptype_str = "any"
                else:
                    porigin = get_origin(pann)
                    pargs = get_args(pann)
                    if porigin is Union:
                        non_none = [a for a in pargs if a is not type(None)]
                        if (
                            len(non_none) == 1
                            and len(pargs) == 2
                            and type(None) in pargs
                        ):
                            inner = non_none[0]
                            try:
                                ptype_str = f"Optional[{inner.__name__}]"
                            except Exception:
                                ptype_str = f"Optional[{str(inner)}]"
                        else:
                            parts = []
                            for a in pargs:
                                try:
                                    parts.append(a.__name__)
                                except Exception:
                                    parts.append(str(a))
                            ptype_str = " or ".join(parts)
                    elif porigin in (list,):
                        try:
                            inner = pargs[0] if pargs else None
                            if inner is None:
                                ptype_str = "List[any]"
                            else:
                                try:
                                    ptype_str = f"List[{inner.__name__}]"
                                except Exception:
                                    ptype_str = f"List[{str(inner)}]"
                        except Exception:
                            ptype_str = "List[any]"
                    elif porigin in (dict,):
                        try:
                            k = pargs[0] if pargs else None
                            v = pargs[1] if len(pargs) > 1 else None
                            k_str = getattr(k, "__name__", str(k)) if k else "any"
                            v_str = getattr(v, "__name__", str(v)) if v else "any"
                            ptype_str = f"Dict[{k_str}, {v_str}]"
                        except Exception:
                            ptype_str = "Dict[any, any]"
                    elif porigin in (tuple,):
                        try:
                            inner = (
                                ", ".join(getattr(a, "__name__", str(a)) for a in pargs)
                                if pargs
                                else ""
                            )
                            ptype_str = f"Tuple[{inner}]"
                        except Exception:
                            ptype_str = "Tuple"
                    else:
                        try:
                            ptype_str = pann.__name__
                        except Exception:
                            ptype_str = str(pann)

                default = None if p.default is inspect._empty else p.default
                required = p.default is inspect._empty and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                doc_entry = args_doc_map.get(pname, {})
                params.append(
                    {
                        "name": pname,
                        "type": None if ptype_str == "any" else ptype_str,
                        "default": default,
                        "required": required,
                        "desc": doc_entry.get("desc", ""),
                    }
                )

            # Returns merge
            ret_ann = sig.return_annotation
            if ret_ann is inspect._empty:
                ret_ann_str = "any"
            else:
                ro = get_origin(ret_ann)
                ra = get_args(ret_ann)
                if ro is Union:
                    non_none = [a for a in ra if a is not type(None)]
                    if len(non_none) == 1 and len(ra) == 2 and type(None) in ra:
                        inner = non_none[0]
                        try:
                            ret_ann_str = f"Optional[{inner.__name__}]"
                        except Exception:
                            ret_ann_str = f"Optional[{str(inner)}]"
                    else:
                        parts = []
                        for a in ra:
                            try:
                                parts.append(a.__name__)
                            except Exception:
                                parts.append(str(a))
                        ret_ann_str = " or ".join(parts)
                elif ro in (list,):
                    try:
                        inner = ra[0] if ra else None
                        if inner is None:
                            ret_ann_str = "List[any]"
                        else:
                            try:
                                ret_ann_str = f"List[{inner.__name__}]"
                            except Exception:
                                ret_ann_str = f"List[{str(inner)}]"
                    except Exception:
                        ret_ann_str = "List[any]"
                elif ro in (dict,):
                    try:
                        k = ra[0] if ra else None
                        v = ra[1] if len(ra) > 1 else None
                        k_str = getattr(k, "__name__", str(k)) if k else "any"
                        v_str = getattr(v, "__name__", str(v)) if v else "any"
                        ret_ann_str = f"Dict[{k_str}, {v_str}]"
                    except Exception:
                        ret_ann_str = "Dict[any, any]"
                elif ro in (tuple,):
                    try:
                        inner = (
                            ", ".join(getattr(a, "__name__", str(a)) for a in ra)
                            if ra
                            else ""
                        )
                        ret_ann_str = f"Tuple[{inner}]"
                    except Exception:
                        ret_ann_str = "Tuple"
                else:
                    try:
                        ret_ann_str = ret_ann.__name__
                    except Exception:
                        ret_ann_str = str(ret_ann)

            if not m_returns:
                m_returns = {
                    "type": None if ret_ann_str == "any" else ret_ann_str,
                    "desc": "",
                }
            else:
                if m_returns.get("type") is None and ret_ann_str != "any":
                    m_returns["type"] = ret_ann_str

            methods.append(
                {
                    "name": name,
                    "kind": kind,
                    "async": inspect.iscoroutinefunction(func),
                    "signature": str(sig),
                    "summary": m_summary,
                    "description": m_description,
                    "parameters": params,
                    "returns": m_returns,
                    "raises": m_raises,
                    "examples": m_examples,
                }
            )

    # Sort methods
    if sort_methods == "name":
        methods.sort(key=lambda m: m["name"])
    elif sort_methods == "kind":
        order = {"instance": 0, "class": 1, "static": 2}
        methods.sort(key=lambda m: (order.get(m["kind"], 99), m["name"]))

    # ----- Render class Markdown -----
    lines.append(f"{h1} Class `{obj_name}`")
    lines.append("")
    lines.append(f"- Module: `{obj_module}`")
    lines.append(f"- Qualname: `{obj_qualname}`")
    lines.append("")

    if "summary" in sec and cls_summary:
        lines.append(f"{h2} Summary")
        lines.append(cls_summary)
        lines.append("")

    if "description" in sec and cls_description:
        lines.append(f"{h2} Description")
        lines.append(cls_description)
        lines.append("")

    if "attributes" in sec and attrs:
        lines.append(f"{h2} Attributes")
        lines.append("")
        if render_tables:
            lines.append("| Name | Type | Description |")
            lines.append("| ---- | ---- | ----------- |")
            for a in attrs:
                typ = a.get("type") or ""
                desc = a.get("desc") or ""
                lines.append(f"| `{a['name']}` | `{typ}` | {desc} |")
        else:
            for a in attrs:
                typ = f" ({a.get('type')})" if a.get("type") else ""
                desc = f": {a.get('desc')}" if a.get("desc") else ""
                lines.append(f"- `{a['name']}`{typ}{desc}")
        lines.append("")

    if "examples" in sec and cls_examples:
        lines.append(f"{h2} Examples")
        lines.append("\n".join(cls_examples))
        lines.append("")

    if "methods" in sec and methods:
        lines.append(f"{h2} Methods")
        lines.append("")
        for m in methods:
            lines.append(f"{h3} `{m['name']}`")
            lines.append("")
            lines.append(f"- Kind: `{m['kind']}`")
            lines.append(f"- Async: `{'true' if m['async'] else 'false'}`")
            if include_signature:
                lines.append("- Signature:")
                lines.append("")
                lines.append("```python")
                lines.append(f"def {m['name']}{m['signature']}")
                lines.append("```")
                lines.append("")
            if m["summary"]:
                lines.append("**Summary**")
                lines.append("")
                lines.append(m["summary"])
                lines.append("")
            if m["description"]:
                lines.append("**Description**")
                lines.append("")
                lines.append(m["description"])
                lines.append("")

            if "parameters" in sec and m["parameters"]:
                lines.append("**Parameters**")
                lines.append("")
                if render_tables:
                    lines.append("| Name | Type | Required | Default | Description |")
                    lines.append("| ---- | ---- | -------- | ------- | ----------- |")
                    for p in m["parameters"]:
                        typ = p["type"] or "any"
                        req = "yes" if p["required"] else "no"
                        default = (
                            "" if p["default"] is None else f"`{repr(p['default'])}`"
                        )
                        desc = p["desc"] or ""
                        lines.append(
                            f"| `{p['name']}` | `{typ}` | {req} | {default} | {desc} |"
                        )
                    lines.append("")
                else:
                    for p in m["parameters"]:
                        typ = p["type"] or "any"
                        req = "required" if p["required"] else "optional"
                        default = (
                            ""
                            if p["default"] is None
                            else f" (default={repr(p['default'])})"
                        )
                        desc = f" - {p['desc']}" if p["desc"] else ""
                        lines.append(f"- `{p['name']}`: {typ}, {req}{default}{desc}")
                    lines.append("")

            if "returns" in sec and m["returns"]:
                rtyp = m["returns"].get("type") or "any"
                rdesc = m["returns"].get("desc") or ""
                lines.append("**Returns**")
                lines.append("")
                lines.append(f"- Type: `{rtyp}`")
                if rdesc:
                    lines.append(f"- Description: {rdesc}")
                lines.append("")

            if "raises" in sec and m["raises"]:
                if m["raises"]:
                    lines.append("**Raises**")
                    lines.append("")
                    for r in m["raises"]:
                        lines.append(f"- `{r['type']}`: {r['desc']}")
                    lines.append("")

            if "examples" in sec and m["examples"]:
                if m["examples"]:
                    lines.append("**Examples**")
                    lines.append("")
                    for ex in m["examples"]:
                        lines.append("```python")
                        lines.append(textwrap.dedent(ex).strip("\n"))
                        lines.append("```")
                        lines.append("")

        lines.append("")

    markdown = "\n".join(lines).rstrip() + "\n"
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
    return markdown


def _generate_function_usage(
    target: object,
    output_path: Optional[str] = None,
    *,
    # Rendering controls
    render_tables: bool = True,
    include_signature: bool = True,
    include_sections: Optional[
        Literal[
            "summary",
            "description",
            "attributes",
            "methods",
            "parameters",
            "returns",
            "raises",
            "examples",
        ]
    ] = None,
    heading_level: int = 2,
) -> str:
    """Internal function for generating usage documentation."""
    # Validate/normalize rendering controls (keep original behavior)
    allowed_sections = {
        "summary",
        "description",
        "attributes",
        "methods",
        "parameters",
        "returns",
        "raises",
        "examples",
    }
    sec = set(include_sections) if include_sections else allowed_sections
    sec = sec.intersection(allowed_sections)
    base_h = max(1, int(heading_level))
    h1 = "#" * base_h
    h2 = "#" * (base_h + 1)

    func = target
    if inspect.isclass(func) or not callable(func):
        raise TypeError("target must be a function/callable (not a class)")

    obj_module = getattr(func, "__module__", "")
    obj_qualname = getattr(func, "__qualname__", getattr(func, "__name__", str(func)))
    obj_name = getattr(func, "__name__", str(func))

    sig = inspect.signature(func)

    # Parse function docstring
    f_raw = inspect.getdoc(func)
    f_summary = ""
    f_description = ""
    f_args = []
    f_returns = None
    f_raises = []
    f_examples = []
    if f_raw:
        fd = textwrap.dedent(f_raw).strip("\n")
        FL = fd.splitlines()
        fn = len(FL)
        fi = 0
        f_sum_lines = []
        while fi < fn and FL[fi].strip():
            f_sum_lines.append(FL[fi].strip())
            fi += 1
        f_summary = " ".join(f_sum_lines).strip()
        while fi < fn and not FL[fi].strip():
            fi += 1
        f_section = None
        f_buffer = {
            "description": [],
            "Args": [],
            "Returns": [],
            "Raises": [],
            "Attributes": [],
            "Examples": [],
        }
        f_headers = set(f_buffer.keys())
        while fi < fn:
            fl = FL[fi]
            fs = fl.strip()
            if fs.endswith(":"):
                fh = fs[:-1]
                if fh in f_headers:
                    f_section = fh
                    fi += 1
                    continue
            if f_section:
                f_buffer[f_section].append(fl)
            else:
                f_buffer["description"].append(fl)
            fi += 1
        f_description = "\n".join([x.rstrip() for x in f_buffer["description"]]).strip()

        # Args
        current = None
        for b in f_buffer["Args"]:
            if not b.strip():
                continue
            mm = re.match(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\(([^)]*)\))?\s*:\s*(.*)$", b
            )
            if mm:
                an = mm.group(1)
                at = mm.group(3)
                ad = mm.group(4).strip()
                current = {"name": an, "type": (at.strip() if at else None), "desc": ad}
                f_args.append(current)
            else:
                if current:
                    current["desc"] += " " + b.strip()

        # Returns
        ret_lines = [l for l in f_buffer["Returns"] if l.strip()]
        if ret_lines:
            mmr = re.match(r"^\s*([^:]+?)\s*:\s*(.*)$", ret_lines[0].strip())
            if mmr:
                rt = mmr.group(1).strip()
                rd = mmr.group(2).strip()
                extra = " ".join(l.strip() for l in ret_lines[1:]).strip()
                if extra:
                    rd = (rd + " " + extra).strip()
                f_returns = {"type": rt, "desc": rd}
            else:
                rd = " ".join(l.strip() for l in ret_lines).strip()
                f_returns = {"type": None, "desc": rd}

        # Raises
        current = None
        for b in f_buffer["Raises"]:
            if not b.strip():
                continue
            rm = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(.*)$", b)
            if rm:
                et = rm.group(1)
                ed = rm.group(2).strip()
                current = {"type": et, "desc": ed}
                f_raises.append(current)
            else:
                if current:
                    current["desc"] += " " + b.strip()

        # Examples
        if f_buffer["Examples"]:
            ex_blocks = []
            block = []
            for l in f_buffer["Examples"]:
                if l.strip():
                    block.append(l.rstrip())
                else:
                    if block:
                        ex_blocks.append("\n".join(block))
                        block = []
            if block:
                ex_blocks.append("\n".join(block))
            f_examples = ex_blocks

    # Build parameters list (merge annotation + doc)
    args_doc_map = {a["name"]: a for a in f_args}
    f_params = []
    for pname, p in sig.parameters.items():
        pann = p.annotation
        if pann is inspect._empty:
            ptype_str = "any"
        else:
            porigin = get_origin(pann)
            pargs = get_args(pann)
            if porigin is Union:
                non_none = [a for a in pargs if a is not type(None)]
                if len(non_none) == 1 and len(pargs) == 2 and type(None) in pargs:
                    inner = non_none[0]
                    try:
                        ptype_str = f"Optional[{inner.__name__}]"
                    except Exception:
                        ptype_str = f"Optional[{str(inner)}]"
                else:
                    parts = []
                    for a in pargs:
                        try:
                            parts.append(a.__name__)
                        except Exception:
                            parts.append(str(a))
                    ptype_str = " or ".join(parts)
            elif porigin in (list,):
                try:
                    inner = pargs[0] if pargs else None
                    if inner is None:
                        ptype_str = "List[any]"
                    else:
                        try:
                            ptype_str = f"List[{inner.__name__}]"
                        except Exception:
                            ptype_str = f"List[{str(inner)}]"
                except Exception:
                    ptype_str = "List[any]"
            elif porigin in (dict,):
                try:
                    k = pargs[0] if pargs else None
                    v = pargs[1] if len(pargs) > 1 else None
                    k_str = getattr(k, "__name__", str(k)) if k else "any"
                    v_str = getattr(v, "__name__", str(v)) if v else "any"
                    ptype_str = f"Dict[{k_str}, {v_str}]"
                except Exception:
                    ptype_str = "Dict[any, any]"
            elif porigin in (tuple,):
                try:
                    inner = (
                        ", ".join(getattr(a, "__name__", str(a)) for a in pargs)
                        if pargs
                        else ""
                    )
                    ptype_str = f"Tuple[{inner}]"
                except Exception:
                    ptype_str = "Tuple"
            else:
                try:
                    ptype_str = pann.__name__
                except Exception:
                    ptype_str = str(pann)

        default = None if p.default is inspect._empty else p.default
        required = p.default is inspect._empty and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        doc_entry = args_doc_map.get(pname, {})
        f_params.append(
            {
                "name": pname,
                "type": None if ptype_str == "any" else ptype_str,
                "default": default,
                "required": required,
                "desc": doc_entry.get("desc", ""),
            }
        )

    # Returns merge
    ret_ann = sig.return_annotation
    if ret_ann is inspect._empty:
        ret_ann_str = "any"
    else:
        ro = get_origin(ret_ann)
        ra = get_args(ret_ann)
        if ro is Union:
            non_none = [a for a in ra if a is not type(None)]
            if len(non_none) == 1 and len(ra) == 2 and type(None) in ra:
                inner = non_none[0]
                try:
                    ret_ann_str = f"Optional[{inner.__name__}]"
                except Exception:
                    ret_ann_str = f"Optional[{str(inner)}]"
            else:
                parts = []
                for a in ra:
                    try:
                        parts.append(a.__name__)
                    except Exception:
                        parts.append(str(a))
                ret_ann_str = " or ".join(parts)
        elif ro in (list,):
            try:
                inner = ra[0] if ra else None
                if inner is None:
                    ret_ann_str = "List[any]"
                else:
                    try:
                        ret_ann_str = f"List[{inner.__name__}]"
                    except Exception:
                        ret_ann_str = f"List[{str(inner)}]"
            except Exception:
                ret_ann_str = "List[any]"
        elif ro in (dict,):
            try:
                k = ra[0] if ra else None
                v = ra[1] if len(ra) > 1 else None
                k_str = getattr(k, "__name__", str(k)) if k else "any"
                v_str = getattr(v, "__name__", str(v)) if v else "any"
                ret_ann_str = f"Dict[{k_str}, {v_str}]"
            except Exception:
                ret_ann_str = "Dict[any, any]"
        elif ro in (tuple,):
            try:
                inner = (
                    ", ".join(getattr(a, "__name__", str(a)) for a in ra) if ra else ""
                )
                ret_ann_str = f"Tuple[{inner}]"
            except Exception:
                ret_ann_str = "Tuple"
        else:
            try:
                ret_ann_str = ret_ann.__name__
            except Exception:
                ret_ann_str = str(ret_ann)

    if not f_returns:
        f_returns = {"type": None if ret_ann_str == "any" else ret_ann_str, "desc": ""}
    else:
        if f_returns.get("type") is None and ret_ann_str != "any":
            f_returns["type"] = ret_ann_str

    # ----- Render function Markdown -----
    lines: list[str] = []
    lines.append(f"{h1} Function `{obj_name}`")
    lines.append("")
    lines.append(f"- Module: `{obj_module}`")
    lines.append(f"- Qualname: `{obj_qualname}`")
    lines.append("")

    if include_signature:
        lines.append(f"{h2} Signature")
        lines.append("")
        lines.append("```python")
        lines.append(f"def {obj_name}{sig}")
        lines.append("```")
        lines.append("")

    if "summary" in sec and f_summary:
        lines.append(f"{h2} Summary")
        lines.append(f_summary)
        lines.append("")

    if "description" in sec and f_description:
        lines.append(f"{h2} Description")
        lines.append(f_description)
        lines.append("")

    if "parameters" in sec and f_params:
        lines.append(f"{h2} Parameters")
        lines.append("")
        if render_tables:
            lines.append("| Name | Type | Required | Default | Description |")
            lines.append("| ---- | ---- | -------- | ------- | ----------- |")
            for p in f_params:
                typ = p["type"] or "any"
                req = "yes" if p["required"] else "no"
                default = "" if p["default"] is None else f"`{repr(p['default'])}`"
                desc = p["desc"] or ""
                lines.append(
                    f"| `{p['name']}` | `{typ}` | {req} | {default} | {desc} |"
                )
            lines.append("")
        else:
            for p in f_params:
                typ = p["type"] or "any"
                req = "required" if p["required"] else "optional"
                default = (
                    "" if p["default"] is None else f" (default={repr(p['default'])})"
                )
                desc = f" - {p['desc']}" if p["desc"] else ""
                lines.append(f"- `{p['name']}`: {typ}, {req}{default}{desc}")
            lines.append("")

    if "returns" in sec and f_returns:
        rtyp = f_returns.get("type") or "any"
        rdesc = f_returns.get("desc") or ""
        lines.append(f"{h2} Returns")
        lines.append("")
        lines.append(f"- Type: `{rtyp}`")
        if rdesc:
            lines.append(f"- Description: {rdesc}")
        lines.append("")

    if "raises" in sec and f_raises:
        if f_raises:
            lines.append(f"{h2} Raises")
            lines.append("")
            for r in f_raises:
                lines.append(f"- `{r['type']}`: {r['desc']}")
            lines.append("")

    if "examples" in sec and f_examples:
        if f_examples:
            lines.append(f"{h2} Examples")
            lines.append("")
            for ex in f_examples:
                lines.append("```python")
                lines.append(textwrap.dedent(ex).strip("\n"))
                lines.append("```")
                lines.append("")

    markdown = "\n".join(lines).rstrip() + "\n"
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
    return markdown


def _generate_usage(
    target: object,
    output_path: Optional[str] = None,
    *,
    # Class-scope controls
    include_private: bool = False,
    include_inherited: bool = False,
    include_properties: bool = True,
    include_methods: bool = True,
    method_kinds: tuple[str, ...] = ("instance", "class", "static"),
    method_include: Optional[list[str]] = None,
    method_exclude: Optional[list[str]] = None,
    attribute_include: Optional[list[str]] = None,
    attribute_exclude: Optional[list[str]] = None,
    # Rendering controls
    sort_methods: Literal["name", "kind", "none"] = "name",
    render_tables: bool = True,
    include_signature: bool = True,
    include_sections: Optional[
        Literal[
            "summary",
            "description",
            "attributes",
            "methods",
            "parameters",
            "returns",
            "raises",
            "examples",
        ]
    ] = None,
    heading_level: int = 2,
) -> str:
    """Generate usage documentation for a class or callable.

    Internal function for auto-generating markdown documentation.
    """
    is_class = inspect.isclass(target)
    is_callable = callable(target)
    if not is_class and not is_callable:
        raise TypeError("target must be a class or a function/callable")

    if is_class:
        return _generate_class_usage(
            target,
            output_path=output_path,
            include_private=include_private,
            include_inherited=include_inherited,
            include_properties=include_properties,
            include_methods=include_methods,
            method_kinds=method_kinds,
            method_include=method_include,
            method_exclude=method_exclude,
            attribute_include=attribute_include,
            attribute_exclude=attribute_exclude,
            sort_methods=sort_methods,
            render_tables=render_tables,
            include_signature=include_signature,
            include_sections=include_sections,
            heading_level=heading_level,
        )

    return _generate_function_usage(
        target,
        output_path=output_path,
        render_tables=render_tables,
        include_signature=include_signature,
        include_sections=include_sections,
        heading_level=heading_level,
    )


def google_search(
    query: str,
    location: Optional[
        Literal["China", "United States", "Germany", "France"]
    ] = "China",
    country: str = "cn",
    language: str = "zh-cn",
    to_be_searched: Optional[str] = None,
    start: str = "1",
    num: str = "10",
):
    """
    Google search page result tool. When asked about a question, you can use this tool to get an original google search page result.
    After browsing the search page result, you can pick some of the valuable result links to view by the `read_url` tool.

    Args:
        query (str): Parameter defines the query you want to search.
            You can use anything that you would use in a regular Google search. e.g. inurl:, site:, intitle:.
            We also support advanced search query parameters such as as_dt and as_eq.
        location (str): Parameter defines from where you want the search to originate.
            If several locations match the location requested, we'll pick the most popular one.
            If location is omitted, the search may take on the location of the proxy.
            When only the location parameter is set, Google may still take into account the proxy’s country, which can influence some results.
            For more consistent country-specific filtering, use the `country` parameter alongside location.
        country (str): Parameter defines the country to use for the Google search.
            It's a two-letter country code. (e.g., cn for China, us for the United States, uk for United Kingdom, or fr for France).
            Your country code should be supported by Google countries codes.
        language (str): Parameter defines the language to use for the Google search.
            It's a two-letter language code. (e.g., zh-cn for Chinese(Simplified), en for English, es for Spanish, or fr for French).
            Your language code should be supported by Google languages.
        to_be_searched (str): parameter defines advanced search parameters that aren't possible in the regular query field.
            (e.g., advanced search for patents, dates, news, videos, images, apps, or text contents).
        start (str): Parameter defines the result offset. It skips the given number of results.
            It's used for pagination. (e.g., 0 (default) is the first page of results, 10 is the 2nd page of results, 20 is the 3rd page of results, etc.).
            Google Local Results only accepts multiples of 20 (e.g. 20 for the second page results, 40 for the third page results, etc.) as the start value.
        num (str): Parameter defines the maximum number of results to return.
            (e.g., 10 (default) returns 10 results, 40 returns 40 results, and 100 returns 100 results).
            The use of num may introduce latency, and/or prevent the inclusion of specialized result types.
            It is better to omit this parameter unless it is strictly necessary to increase the number of results per page.
            Results are not guaranteed to have the number of results specified in num.

    Return:
        (str) The search report in markdown format.
    """

    try:
        import serpapi
    except ImportError as e:
        return 'No web search backend found, install by `pip install "parquool[websearch]"`'

    api_keys = [key.strip() for key in os.getenv("SERPAPI_KEY").split(",")]
    random.shuffle(api_keys)
    param_names = ["location", "gl", "hl", "tbs", "start", "num"]
    param_vars = [location, country, language, to_be_searched, start, num]
    params = {"engine": "google", "q": query}
    for name, param in zip(param_names, param_vars):
        if param is not None:
            params[name] = param

    for api_key in api_keys:
        try:
            result = serpapi.search(params, api_key=api_key)
        except Exception as e:
            continue
        result = result.as_dict()
        break

    else:
        return f"Request failed, please check your parameter and try again."

    search_metadata = result["search_metadata"]
    search_report = (
        f"# Search Report for Query {query}\n\n"
        f"Created at {search_metadata['created_at']}, processed at {search_metadata['processed_at']}."
        f"You can get access to json file at {search_metadata['json_endpoint']}, html file at {search_metadata['raw_html_file']}."
        f"{result['search_information']['total_results']} Found. {search_metadata['total_time_taken']} seconds taken.\n\n"
    )
    for ores in result.get("organic_results", []):
        search_report += (
            f"## [{ores['title']}]({ores['link']})\n\n"
            + f"[{ores['source']}] {ores['snippet']}\n\n"
            + (f"> date: {ores['date']}\n" if "date" in ores.keys() else "")
        )
    return search_report


def read_url(
    url_or_urls: Union[str, List],
    engine: Literal["direct", "browser"] = "browser",
    return_format: Literal["markdown", "html", "text", "screeshot"] = "markdown",
    with_links_summary: Literal["all", "true"] = "true",
    with_image_summary: Literal["all", "true"] = "true",
    retain_image: bool = False,
    do_not_track: bool = True,
    set_cookie: Optional[str] = None,
    max_length_each: int = 100000,
):
    """Fetch and summarize the readable content of one or more URLs via the r.jina.ai reader proxy.

    The agent should call this tool when it needs the actual page text or a snapshot of the page to
    extract facts, quotes, or to decide whether the page is worth further processing.

    Args:
        url_or_urls (Union[str, List]): A single URL string or a list of URL strings to read.
            Provide full URLs as produced by search results (e.g., "https://example.com/page").
        engine (Literal["direct", "browser"], optional): Which fetching engine the proxy
            should use. "direct" performs a direct HTTP fetch, "browser" uses a headless
            browser to render the page (recommended for JS-heavy sites). If omitted, the
            proxy service default is used.
        return_format (Literal["markdown", "html", "text", "screeshot"], optional):
            Desired format of the proxy's returned content:
            - "markdown": proxy attempts to extract and return a clean Markdown version.
            - "html": returns raw or minimally processed HTML.
            - "text": plain text extraction.
            - "screeshot": request an image capture of the page (note the implementation
            currently expects the literal "screeshot").
            If omitted, the proxy service default is used.
        with_links_summary (Literal["all", "true"], optional):
            Wether to summarize all the links in the end of the result page:
            - "all": list all the links in the page and summarize them in the end.
            - "true": list all the unique links in the page and summarize them in the end.
            - None: keep links in-line in result.
        with_image_summary (Literal["all", "true"], optional):
            Wether to summarize all the images in the end of the result page:
            - "all": list all the images in the page and summarize them in the end.
            - "true": list all the unique images in the page and summarize them in the end.
            - None: keep images in-line in result.
        retain_image (bool, optional): If True (default), the returned HTML/Markdown may
            include image references. If False, images are disabled/removed by the proxy.
        do_not_track (bool, optional): If True (default), the header DNT: 1 is sent to
            indicate "do not track" preference to the proxy.
        set_cookie (str, optional): If provided, sets a Cookie header value to be passed
            to the proxy (useful for accessing pages that require a specific cookie).
        max_length_each (int, optional): Maximum number of characters to include from each
            successful response in the returned report. Defaults to 7168. Longer pages will
            be truncated to this length.

    Returns:
        str: A Markdown-formatted report string describing the results for each requested URL.
        The report contains:
        - A summary header with the number of input URLs.
        - "Success Requests" section listing each successful URL and the first
            max_length_each characters of the returned content.
        - "Failure Requests" section listing each URL that failed and the associated
            error message.
    """
    urls = url_or_urls if isinstance(url_or_urls, list) else [url_or_urls]
    headers = {}
    if engine:
        headers["X-Engine"] = engine
    if return_format:
        headers["X-Return-Format"] = return_format
    if with_links_summary:
        headers["X-With-Links-Summary"] = with_links_summary
    if with_image_summary:
        headers["X-With-Images-Summary"] = with_image_summary
    if not retain_image:
        headers["X-Retain-Images"] = "none"
    if do_not_track:
        headers["DNT"] = "1"
    if set_cookie:
        headers["X-Set-Cookie"] = set_cookie
    failure = []
    success = []

    for url in urls:
        try:
            response = requests.get(f"https://r.jina.ai/{url}", headers=headers)
            response.raise_for_status()
            success.append((url, response.text[:max_length_each]))
        except Exception as e:
            failure.append((url, str(e)))

    read_report = f"# Read results for {len(urls)}\n\n"
    read_report += "## Success Resquests\n\n"
    for suc in success:
        read_report += f"### {suc[0]}\n\n{suc[1]}"

    read_report += "## Failure Resquests\n\n"
    for fai in failure:
        read_report += f"### {fai[0]}\n\n{fai[1]}"

    return read_report
