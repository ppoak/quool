import re
import logging
import imaplib
import smtplib
import markdown
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from email import message_from_bytes
from email.mime.text import MIMEText
from email.header import decode_header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


def setup_logger(
    name, 
    file: str | Path = None, 
    stream: bool = True, 
    level: str = logging.INFO, 
    style: int = 1,
    clear: bool = False,
    rotation: str = None, 
    max_bytes: int = None, 
    backup_count: int = None, 
    when: str = None, 
    interval: int = None,
):
    """
    Set up a logger with specified file and stream handlers, with various formatting styles and optional log rotation.

    Args:
        name (str): Name of the logger.
        file (str or Path, optional): File path for file handler. If None, no file output is used.
        stream (bool, optional): Whether to add a stream handler for console output. Defaults to True.
        level (int, optional): Logging level. Defaults to logging.INFO.
        style (int, optional): Integer representing the desired logging style for formatting output.
                               Style 1: Basic info
                               Style 2: Medium info
                               Style 3: Detailed info
                               Style 4: Full info
        rotation (str, optional): Type of log rotation. Options are:
                                  - 'size': Rotate based on file size.
                                  - 'time': Rotate based on time intervals.
                                  - None: No rotation (default).
        max_bytes (int, optional): Maximum size of a log file before rotation (only for 'size' rotation).
        backup_count (int, optional): Number of backup files to keep (default: 5 for 'size', 7 for 'time').
        when (str, optional): Time unit for rotation (e.g., 'midnight', 'H') (only for 'time' rotation).
        interval (int, optional): Interval for time-based rotation (default: 1, only for 'time' rotation).

    Returns:
        logging.Logger: Configured logger.
    """
    if file and clear:
        Path(file).write_text("")
    
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Avoid adding handlers multiple times
    
    logger.setLevel(level)

    # Define formatter styles
    formatter_styles = {
        1: logging.Formatter("[%(levelname)s]@[%(asctime)s]-[%(name)s]: %(message)s"),
        2: logging.Formatter("[%(levelname)s]@[%(asctime)s]-[%(name)s]@[%(module)s:%(lineno)d]: %(message)s"),
        3: logging.Formatter("[%(levelname)s]@[%(asctime)s]-[%(name)s]@[%(module)s:%(lineno)d#%(funcName)s]: %(message)s"),
        4: logging.Formatter("[%(levelname)s]@[%(asctime)s]-[%(name)s]@[%(module)s:%(lineno)d#%(funcName)s~%(process)d:%(threadName)s]: %(message)s")
    }

    # Select the formatter based on the style parameter
    formatter = formatter_styles.get(style, formatter_styles[1])  # Default to style 1

    # Add stream handler if specified
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add file handler if specified
    if file:
        if rotation == 'size':
            # Use size-based rotation
            file_handler = RotatingFileHandler(
                file,
                maxBytes=max_bytes or 10 * 1024 * 1024,  # Default to 10 MB
                backupCount=backup_count or 5  # Default to 5 backups
            )
        elif rotation == 'time':
            # Use time-based rotation
            file_handler = TimedRotatingFileHandler(
                file,
                when=when or 'midnight',  # Default to midnight
                interval=interval or 1,  # Default to 1 day
                backupCount=backup_count or 7  # Default to 7 backups
            )
        else:
            # No rotation
            file_handler = logging.FileHandler(file)

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def evaluate(
    value: pd.Series,
    benchmark: pd.Series = None,
    turnover: pd.Series = None,
    trades: pd.DataFrame = None,
):
    """Evaluate the performance of a trading strategy.

    Args:
        value (pd.Series): The value of the portfolio.
        benchmark (pd.Series, optional): The value of the benchmark. Defaults to None.
        turnover (pd.Series, optional): The turnover rate. Defaults to None.
        trades (pd.Series): The trading records.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    net_value = value / value.iloc[0]
    returns = net_value.pct_change().fillna(0)
    drawdown = net_value / net_value.cummax() - 1
    max_drawdown = drawdown.min()
    max_drawdown_end = drawdown.idxmin()
    max_drawdown_start = drawdown.loc[:max_drawdown_end][drawdown.loc[:max_drawdown_end] == 0].index[-1]

    # Benchmark Comparison Metrics
    if benchmark is None:
        benchmark = pd.Series(np.ones_like(net_value), index=net_value.index)
    benchmark_returns = benchmark.pct_change().fillna(0)
    excess_returns = returns - benchmark_returns

    evaluation = pd.Series(name="evaluation")
    # Basic Performance Metrics
    evaluation["total_return(%)"] = (net_value.iloc[-1] - 1) * 100
    evaluation["annual_return(%)"] = (
        (1 + evaluation["total_return(%)"] / 100) ** (365 / (net_value.index[-1] - net_value.index[0]).days) - 1
    ) * 100
    evaluation["annual_volatility(%)"] = (returns.std() * np.sqrt(252)) * 100
    evaluation["sharpe_ratio"] = (
        evaluation["annual_return(%)"] / evaluation["annual_volatility(%)"]
        if evaluation["annual_volatility(%)"] != 0
        else np.nan
    )
    evaluation["calmar_ratio"] = (
        evaluation["annual_return(%)"] / abs(max_drawdown * 100)
        if max_drawdown != 0
        else np.nan
    )
    downside_std = returns[returns < 0].std()
    evaluation["sortino_ratio(%)"] = (
        evaluation["annual_return(%)"] / (downside_std * np.sqrt(252))
        if downside_std != 0
        else np.nan
    )

    # Risk Metrics
    evaluation["max_drawdown(%)"] = max_drawdown * 100
    evaluation["max_drawdown_period"] = max_drawdown_end - max_drawdown_start
    var_95 = np.percentile(returns, 5) * 100
    evaluation["VaR_5%(%)"] = var_95
    cvar_95 = returns[returns <= var_95 / 100].mean() * 100
    evaluation["CVaR_5%(%)"] = cvar_95

    # Turnover Ratio
    if turnover is not None:
        evaluation["turnover_ratio(%)"] = turnover.mean() * 100
    else:
        evaluation["turnover_ratio(%)"] = np.nan

    # Alpha and Beta, Benchmark related
    beta = returns.cov(benchmark_returns) / benchmark_returns.var() if benchmark_returns.var() != 0 else np.nan
    evaluation["beta"] = beta
    evaluation["alpha(%)"] = (
        (returns.mean() - beta * benchmark_returns.mean()) * 252 * 100
        if beta is not np.nan
        else np.nan
    )
    evaluation["excess_return(%)"] = excess_returns.mean() * 252 * 100
    evaluation["excess_volatility(%)"] = excess_returns.std() * np.sqrt(252) * 100
    tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
    evaluation["information_ratio"] = (
        evaluation["excess_return(%)"] / tracking_error
        if tracking_error != 0
        else np.nan
    )

    # Trading behavior
    if trades is not None:
        evaluation["position_duration(days)"] = trades["duration"].mean()
        profit = trades["close_amount"] - trades["open_amount"]
        evaluation["trade_win_rate(%)"] = profit[profit > 0].count() / profit.count() * 100
        evaluation["trade_return(%)"] = profit.sum() / trades["open_amount"].sum() * 100
    else:
        evaluation["position_duration(days)"] = np.nan
        evaluation["trade_win_rate(%)"] = np.nan
        evaluation["trade_return(%)"] = np.nan

    # Distribution Metrics
    evaluation["skewness"] = returns.skew()
    evaluation["kurtosis"] = returns.kurtosis()
    positive_returns = returns[returns.ge(0 if benchmark is None else benchmark_returns)].count()
    evaluation["day_return_win_rate(%)"] = (positive_returns / returns.count()) * 100
    monthly_returns = net_value.resample("ME").last().pct_change().fillna(0)
    evaluation["monthly_return_std(%)"] = monthly_returns.std() * 100
    evaluation["monthly_win_rate(%)"] = (monthly_returns > 0).sum() / len(monthly_returns) * 100
    return evaluation


class Emailer:
    """A class to create and send emails, supporting Markdown to HTML conversion and embedded images."""

    def __init__(
        self, 
        root_url: str,
        smtp_server: str = None, 
        imap_server: str = None, 
        smtp_port: int = 0,
        imap_port: int = 993,
    ):
        """
        Initializes the EmailSender class.

        Args:
            smtp_server (str): The SMTP server address. Default is 'smtp.163.com'.
            port (int): The SMTP server port. Default is 0.
        """
        if root_url is not None and (smtp_server is not None or imap_server is not None):
            raise ValueError("root_url and smtp_server/imap_server cannot both be specified.")

        self.root_url = root_url
        self.smtp_server = smtp_server or f"smtp.{root_url}"
        self.imap_server = imap_server or f"imap.{root_url}"
        self.smtp_port = smtp_port
        self.imap_port = imap_port
        self.smtp_server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        self.smtp_server.starttls()
        self.imap_server = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
    
    def login(self, address: str, password: str):
        """
        Logs into the SMTP server with the provided credentials.

        Args:
            address (str): Sender's email address.
            password (str): Sender's email account password.
        """
        self.address = address
        self.smtp_server.login(address, password)
        self.imap_server.login(address, password)

    def send(
        self, 
        receivers: str, 
        subject: str, 
        message: str, 
        cc: str = None
    ):
        """
        Sends an email with a Markdown-formatted body, converted to HTML, with embedded images if specified.

        Args:
            receivers (str): The recipient's email address(es), separated by commas if multiple.
            subject (str): The subject of the email.
            message (str): The body of the email in Markdown format.
            cc (str): The CC recipient's email address(es), separated by commas if multiple (optional).
        """
        # Create a multipart message
        msg = MIMEMultipart("related")
        msg["From"] = self.address
        msg["To"] = receivers
        # Add CC recipients to the email header
        if cc:
            msg["Cc"] = cc
        msg["Subject"] = subject

        # Convert Markdown to HTML using the `markdown` package
        html_body = markdown.markdown(
            message,
            extensions=["tables", "fenced_code", "codehilite", "extra"]
        )

        # Find all paths in the markdown using a regular expression
        file_paths = re.findall(r'!\[.*?\]\((.*?)\)', message)

        # Attach images and files as needed
        for i, file_path in enumerate(file_paths):
            file = Path(file_path)
            if file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif'}:
                with file.open("rb") as img:
                    img_data = img.read()
                    # Create a unique content ID
                    cid = f"image{i}"
                    image_mime = MIMEImage(img_data)
                    image_mime.add_header("Content-ID", f"<{cid}>")
                    image_mime.add_header("Content-Disposition", "inline", filename=file.name)
                    msg.attach(image_mime)
                    # Replace the file path in the HTML body with a cid reference
                    html_body = html_body.replace(file_path, f"cid:{cid}")
            else:
                with file.open("rb") as attachment:
                    part = MIMEText(attachment.read(), "base64")
                    part.add_header("Content-Disposition", f"attachment; filename={file.name}")
                    msg.attach(part)

        # Update the HTML part with embedded image references
        msg.attach(MIMEText(html_body, "html"))

        # Prepare the recipient list, including CC recipients
        recipient_list = receivers.split(",")
        if cc:
            recipient_list += cc.split(",")

        # Send the email
        self.smtp_server.sendmail(self.address, recipient_list, msg.as_string())
    
    def receive(
        self, 
        mailbox: str = "INBOX", 
        criteria: str = "ALL", 
        attachments: str | Path = None,
    ):
        """
        Receives emails from the specified mailbox.

        Args:
            mailbox (str): The mailbox to select. Default is 'INBOX'.
            criteria (str): The search criteria to filter emails. Default is 'ALL'.
            attachments (bool): Whether to save email attachments. Default is False.

        Returns:
            list[dict]: A list of dictionaries containing email details (subject, from, date, body, attachments).
        """
        # Select the mailbox
        self.imap_server.select(mailbox)
        # Search emails based on criteria
        status, email_ids = self.imap_server.search(None, criteria)
        if status != "OK":
            raise Exception("Failed to search emails.")

        emails = []
        for email_id in email_ids[0].split():
            # Fetch the email
            status, data = self.imap_server.fetch(email_id, "(RFC822)")
            if status != "OK":
                continue

            raw_email = data[0][1]
            email_message = message_from_bytes(raw_email)
            email_data = {"subject": None, "from": None, "date": None, "body": None, "attachments": []}

            # Decode email subject
            subject = decode_header(email_message["Subject"])[0]
            if isinstance(subject[0], bytes):
                email_data["subject"] = subject[0].decode(subject[1] or "utf-8")
            else:
                email_data["subject"] = subject[0]

            # Decode email sender
            email_data["from"] = email_message.get("From")
            email_data["date"] = email_message.get("Date")

            # Extract email body
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        email_data["body"] = part.get_payload()
                    elif "attachment" in content_disposition and attachments:
                        # Save attachment
                        filename = part.get_filename()
                        if filename:
                            header = decode_header(filename)
                            filename = ''.join([str(part, charset if isinstance(part, bytes) else 'utf-8') for part, charset in header])
                            filename = attachments / filename
                            attachment_data = part.get_payload(decode=True)
                            email_data["attachments"].append({"filename": str(filename), "data": attachment_data})
                            with open(filename, "wb") as f:
                                f.write(attachment_data)
            else:
                email_data["body"] = email_message.get_payload(decode=True).decode()

            emails.append(email_data)

        return emails

    @staticmethod
    def notify(
        address: str,
        password: str,
        receiver: str,
        cc: str = None
    ):
        """
        A decorator to send a notification email after executing a task.

        Args:
            address (str): Sender's email address.
            password (str): Sender's email password.
            receiver (str): The recipient's email address.
            cc (str, optional): CC recipients. Defaults to None.

        Returns:
            callable: A decorated function.
        """
        def decorator(task):
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
                            result_str = result.head().to_markdown() + '\n\n...\n\n' + result.tail().to_markdown()
                        else:
                            result_str = result.to_markdown()
                    elif isinstance(result, dict):
                        result_str = pd.DataFrame(result).to_markdown()
                    else:
                        result_str = str(result)
                    args = [str(arg).replace(">", "&gt;").replace("<", "&lt;") for arg in args]
                    kwargs = {key: str(value).replace(">", "&gt;").replace("<", "&lt;") for key, value in kwargs.items()}
                    message = (
                        f"{result_str}\n\n"
                        f"> *Parameters: {args} {kwargs}*\n\n"
                        f"> *Run from {begin} to {end} ({duration})*"
                    )
                except Exception as e:
                    success = False
                    result = str(e)
                    end = pd.to_datetime("now")
                    args = [str(arg).replace(">", "&gt;").replace("<", "&lt;") for arg in args]
                    kwargs = {key: str(value).replace(">", "&gt;").replace("<", "&lt;") for key, value in kwargs.items()}
                    duration = end - begin
                    message = (
                        f"```\n{'\n'.join([trace.replace('^', '') for trace in traceback.format_exception(type(e), e, e.__traceback__)])}\n```\n\n"
                        f"> *Parameters: {args} {kwargs}*\n\n"
                        f"> *Run from {begin} to {end} ({duration})*"
                    )
                finally:
                    emailer = Emailer(root_url=address.split('@')[-1])
                    subject = f"Task {task.__name__} {'success' if success else 'failure'}"
                    emailer.login(address, password)
                    emailer.send(
                        receivers=receiver,
                        subject=subject,
                        message=message,
                        cc=cc
                    )
                    emailer.close()
                return result
            return wrapper
        return decorator

    def close(self):
        """Closes the SMTP server connection."""
        self.smtp_server.quit()
        self.imap_server.logout()

    def __enter__(self):
        """Enters the runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the runtime context and closes the SMTP connection."""
        self.close()
