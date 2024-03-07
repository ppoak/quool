__version__ = "5.0.1"


import os
import time
import quool
import tarfile
import argparse
import datetime
import pandas as pd
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .base import (
    qtd, qtm, fin, idxwgt, idxqtd,
    idxqtm, sec, ids, con, div, ins, prx,
)


TABLE_DICT = {
    "index-weights": idxwgt,
    "industry-info": ids,
    "instruments-info": ins,
    "quotes-day": qtd,
    "quotes-min": qtm,
    "security-margin": sec,
    "stock-connect": con,
    "financial": fin,
    "dividend-split": div,
    "index-quotes-day": idxqtd,
    "index-quotes-min": idxqtm,
}


def ricequant_fetcher(
    user: str,
    password: str,
    driver: str,
    target: str,
    logfile: str = 'update.log',
):
    logger = quool.Logger("ricequant", stream=False, display_name=True, file=logfile)
    logger.debug("=" * 5 + " ricequant fetcher start " + "=" * 5)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    prefs = {"download.default_directory" : target}
    chrome_options.add_experimental_option("prefs", prefs)

    service = Service(driver)
    driver: webdriver.Chrome = webdriver.Chrome(service=service, options=chrome_options)

    logger.debug("visiting https://www.ricequant.com/")
    driver.get("https://www.ricequant.com/")

    time.sleep(5)
    login_button = driver.find_element(By.CLASS_NAME, "user-status")
    login_button.click()
    logger.debug("loging in")
    password_login = driver.find_element(By.CSS_SELECTOR, '.el-dialog__body > div > div > ul > li:nth-child(2)')
    password_login.click()
    inputs = driver.find_elements(By.CLASS_NAME, 'el-input__inner')
    for ipt in inputs:
        if '邮箱' in ipt.get_attribute('placeholder'):
            account = ipt
        if '密码' in ipt.get_attribute('placeholder'):
            passwd = ipt
    account.send_keys(user)
    passwd.send_keys(password)
    login_button = driver.find_element(By.CSS_SELECTOR, 'button.el-button.common-button.btn--submit')
    login_button.click()
    logger.debug("logged in and redirect to reserch subdomain")

    time.sleep(5)
    driver.get('https://www.ricequant.com/research/')
    time.sleep(5)
    notebook_list = driver.find_element(By.ID, 'notebook_list')
    logger.debug("finding `ricequant_fetcher.ipynb`")
    items = notebook_list.find_elements(By.CSS_SELECTOR, '.list_item.row')
    for item in items:
        if 'ricequant_fetcher' in item.text:
            file = item.find_element(By.CSS_SELECTOR, 'a')
            break
    file.click()

    logger.debug("wait for some time before redirect to `ricequant_fetcher.ipynb`")
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(5)
    cell = driver.find_element(By.CSS_SELECTOR, '#menus > div > div > ul > li:nth-child(5)')
    cell.click()
    runall = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '#run_all_cells'))
    )
    logger.debug("start running")
    runall.click()
    unfinished = 0
    while True:
        prompts = driver.find_elements(By.CSS_SELECTOR, '.prompt.input_prompt')
        unfinished_cur = 0
        for prompt in prompts:
            if '*' in prompt.text:
                unfinished_cur += 1
        if unfinished_cur != unfinished:
            logger.debug(f'tasks left: {unfinished_cur}/{len(prompts)}')
            unfinished = unfinished_cur
        if unfinished == 0:
            break
    logger.debug("all tasks are finished")
    driver.close()
    driver.switch_to.window(driver.window_handles[-1])
    driver.refresh()
    time.sleep(5)
    notebook_list = driver.find_element(By.ID, 'notebook_list')
    items = notebook_list.find_elements(By.CSS_SELECTOR, '.list_item.row')
    todaystr = datetime.datetime.today().strftime(r'%Y%m%d')
    logger.debug("finding the generated data file")
    for item in items:
        if todaystr in item.text and '.tar.gz' in item.text:
            file = item
            break
    file.click()
    filename = file.text.splitlines()[0]
    filepath_parent = Path(target)
    download_button = driver.find_element(By.CSS_SELECTOR, '.download-button.btn.btn-default.btn-xs')
    download_button.click()
    logger.debug(f"downloading {filename}")
    previous_size = -1
    time.sleep(1)
    while True:
        filepath = list(filepath_parent.glob(f'{filename}*'))[0]
        current_size = os.path.getsize(filepath)
        if current_size == previous_size:
            logger.debug(f"{filepath} is finished downloading")
            break
        else:
            previous_size = current_size
            time.sleep(2)
    logger.debug(f"deleting {filename}")
    time.sleep(5)
    delete_button = driver.find_element(By.CSS_SELECTOR, '.delete-button.btn.btn-default.btn-xs.btn-danger')
    delete_button.click()
    time.sleep(5)
    double_check_delete_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.btn.btn-default.btn-sm.btn-danger'))
    )
    double_check_delete_button.click()
    
    driver.quit()
    logger.debug("=" * 5 + " ricequant fetcher stop " + "=" * 5)
    return filepath

def parse_args():
    parser = argparse.ArgumentParser("RiceQuant Automate Updater")
    parser.add_argument('--user', type=str, required=True, help="ricequant user name")
    parser.add_argument('--password', type=str, required=True, help="ricequant password")
    parser.add_argument('--driver', type=str, default="./chromedriver", help="path to your chromedriver")
    parser.add_argument('--target', type=str, default='.', help="path to your target directory")
    parser.add_argument('--backup', type=str, default='.', help="path to your backup directory")
    parser.add_argument('--logfile', type=str, default='./update.log', help="path to your logfile")
    args = parser.parse_args()
    return args

def update_data(filename: str, logfile: str = 'debug.log'):
    logger = quool.Logger("UpdateData", stream=False, display_name=True, file=logfile)
    logger.debug("=" * 5 + " update data start " + "=" * 5)
    data_path = Path(filename).expanduser().resolve()
    directory = data_path.parent / data_path.stem
    directory.mkdir(exist_ok=True, parents=True)
    if not data_path.is_dir():
        with tarfile.open(data_path, f'r:{data_path.suffix[1:]}') as tar:
            tar.extractall(path=directory)
        data_path.unlink()
    else:
        directory = data_path

    logger.debug('-' * 20)
    for file in directory.glob('**/*.parquet'):
        logger.debug(f'processing {file}')
        _update_data(file)

    directory.rmdir()
    logger.debug("=" * 5 + " update data stop " + "=" * 5)

def _update_data(filename: str | Path):
    filename = Path(filename).expanduser().resolve()
    name = filename.stem.split('_')[0]
    table = TABLE_DICT[name]
    df = pd.read_parquet(filename)
    table.update(df)
    filename.unlink()

def update_proxy(logfile: str = 'debug.log'):
    logger = quool.Logger("UpdateProxy", stream=False, display_name=True, file=logfile)
    logger.debug("=" * 5 + " update proxy start " + "=" * 5)
    try:
        prx.add_kuaidaili(pages=10)
    except:
        logger.warning("kuaidaili failed")
    try:
        prx.add_kxdaili(pages=10)
    except:
        logger.warning("kxdaili failed")
    try:
        prx.add_ip3366(pages=10)
    except:
        logger.warning("ip3366 failed")
    try:
        prx.add_89ip(pages=10)
    except:
        logger.warning("89ip failed")
    logger.debug("=" * 5 + " update proxy stop " + "=" * 5)

def backup_data(uri: str | Path, backup: str | Path, logfile: str = "debug.log"):
    logger = quool.Logger("UpdateProxy", stream=False, display_name=True, file=logfile)
    logger.debug("=" * 5 + " backup data start " + "=" * 5)
    uri = Path(uri).expanduser().resolve()
    backup = Path(backup).expanduser().resolve()
    backup.mkdir(parents=True, exist_ok=True)
    if not uri.is_dir():
        raise ValueError('uri must be a directory')
    with tarfile.open(str(backup / uri.name) + '.tar.gz', "w:gz") as tar:
        for file in uri.glob('**/*.parquet'):
            tar.add(file)
    logger.debug("=" * 5 + " backup data stop " + "=" * 5)

def unbackup_data(backup: str | Path, uribase: str | Path = '/'):
    backup = Path(backup).expanduser().resolve()
    uribase = Path(uribase).expanduser().resolve()
    with tarfile.open(backup, f"r:{backup.suffix.split('.')[-1]}") as tar:
        tar.extractall(path=uribase)

if __name__ == "__main__":
    args = parse_args()
    user, password, driver, target, logfile = (args.user, 
        args.password, args.driver, args.target, args.logfile)
    filename = ricequant_fetcher(user, password, driver, target, logfile)
    update_data(filename, logfile=logfile)
    update_proxy(logfile=logfile)
    for uri in Path("./data/").iterdir():
        backup_data(uri, "./data/backupdata")