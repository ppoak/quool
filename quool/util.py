import re
import time
import logging
import smtplib
import requests
import markdown
import traceback
import pandas as pd
from retry import retry
from pathlib import Path
from copy import deepcopy
from email import encoders
from functools import wraps
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


def setup_logger(
    name,
    file: str | Path = None,
    stream: bool = True,
    level: str = logging.INFO,
    style: int | str = 1,
    replace: bool = False,
    clear: bool = False,
    rotation: str = None,
    max_bytes: int = None,
    backup_count: int = None,
    when: str = None,
    interval: int = None,
):
    if file and clear:
        Path(file).write_text("")

    if not replace:
        logger = logging.getLogger(name)
    else:
        logger = logging.Logger(name, level=level)
    if logger.hasHandlers():
        return logger  # Avoid adding handlers multiple times

    logger.setLevel(level)

    # Define formatter styles
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

    # Select the formatter based on the style parameter
    formatter = formatter_styles.get(style, style)  # Not found, use user setting

    # Add stream handler if specified
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add file handler if specified
    if file:
        if rotation == "size":
            # Use size-based rotation
            file_handler = RotatingFileHandler(
                file,
                encoding="utf-8",
                maxBytes=max_bytes or 10 * 1024 * 1024,  # Default to 10 MB
                backupCount=backup_count or 5,  # Default to 5 backups
            )
        elif rotation == "time":
            # Use time-based rotation
            file_handler = TimedRotatingFileHandler(
                file,
                encoding="utf-8",
                when=when or "midnight",  # Default to midnight
                interval=interval or 1,  # Default to 1 day
                backupCount=backup_count or 7,  # Default to 7 backups
            )
        else:
            # No rotation
            file_handler = logging.FileHandler(file, encoding="utf-8")

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def notify_task(
    sender: str,
    password: str,
    receiver: str,
    smtp_server: str,
    cc: str = None,
    smtp_port: int = None,
):
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
                        with file.open("rb") as img:
                            img_data = img.read()
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
                        try:
                            part = MIMEText(file.read_text("utf-8"), "plain", "utf-8")
                            part.add_header(
                                "Content-Disposition",
                                f"attachment; filename={file.name}",
                            )
                            content.attach(part)
                        except UnicodeDecodeError:
                            part = MIMEBase("application", "octet-stream")
                            part.set_payload(file.read_bytes())
                            encoders.encode_base64(part)
                            part.add_header(
                                "Content-Disposition",
                                f"attachment; filename={file.name}",
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
    url: str, method: str = "GET", proxies: dict = None, delay: float = 1, **kwargs
):
    proxies = proxies or []
    if not isinstance(proxies, list):
        proxies = [proxies]
    proxies = deepcopy(proxies)
    for proxy in proxies:
        try:
            response = requests.request(method=method, url=url, proxies=proxy, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            time.sleep(delay)
    else:
        response = requests.request(method=method, url=url, **kwargs)
        response.raise_for_status()
        return response
