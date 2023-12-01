import time
import quool
import argparse
import requests
import subprocess
import pandas as pd
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed


class Checker(quool.Request):

    def __init__(self, proxies: list[dict], url: str = "http://httpbin.org/ip"):
        super().__init__(
            url = [url] * len(proxies),
            proxies = proxies,
            timeout = 2.0,
            retry = 1,
        )

    def _req(self, url: str, proxy: dict):
        method = getattr(requests, self.method)
        retry = self.retry or 1
        for t in range(1, retry + 1):
            try:
                resp = method(
                    url, headers=self.headers, proxies=proxy,
                    timeout=self.timeout, **self.kwargs
                )
                resp.raise_for_status()
                if self.verbose:
                    print(f'[+] {url} try {t}')
                return resp
            except Exception as e:
                if self.verbose:
                    print(f'[-] {e} {url} try {t}')
                time.sleep(self.delay)
        return None

    def request(self) -> list[requests.Response]:
        responses = []
        for proxy, url in zip(self.proxies, self.url):
            resp = self._req(url=url, proxy=proxy)
            responses.append(resp)
        self.responses = responses
        return self
    
    def para_request(self) -> list[requests.Response]:
        self.responses = Parallel(n_jobs=-1, backend='loky')(
            delayed(self._req)(url, proxy) for url, proxy in zip(self.url, self.proxies)
        )
        return self
    
    def callback(self):
        results = []
        for i, res in enumerate(self.responses):
            if res is None:
                continue
            results.append(self.proxies[i])
        return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description='Data Dumping Script')
    parser.add_argument('-f', '--file', dest='filepath', type=str, help='Input the path to the `.tar.gz` file')
    args = parser.parse_args()
    return args

table_dict = {
    "index-weights": partial(quool.AssetTable, date_index="date", code_index="order_book_id"),
    "industry-info": partial(quool.AssetTable, date_index="date", code_index="order_book_id"),
    "instruments-info": quool.FrameTable,
    "quotes-day": partial(quool.AssetTable, date_index="date", code_index="order_book_id"),
    "quotes-min": partial(quool.AssetTable, date_index="datetime", code_index="order_book_id"),
    "security-margin": partial(quool.AssetTable, date_index="date", code_index="order_book_id"),
    "stock-connect": partial(quool.AssetTable, date_index="date", code_index="order_book_id"),
    "financial": partial(quool.DiffTable, date_index="date", code_index="order_book_id"),
    "index-quotes-day": partial(quool.AssetTable, date_index="date", code_index="order_book_id"),
}

def update_data(args):
    if not args.filepath:
        files = list(Path('/home/data/').glob('*.tar.gz')) + list(Path('/home/data/').glob('*.tar'))
        if not files:
            return
        else:
            args.filepath = files[0]
    
    data_path = Path(args.filepath)
    directory = data_path.parent / data_path.stem
    directory.mkdir(exist_ok=True)
    if data_path.suffix == '.tar':
        parameter = '-xvf'
    elif data_path.suffix == '.gz':
        parameter = '-xvzf'
    else:
        raise ValueError('Invalid file extension')
    subprocess.run(['tar', parameter, data_path, "-C", str(directory.resolve())])
    data_path.unlink()
    data_path = data_path.parent / data_path.stem

    print('-' * 20)
    for file in data_path.glob('**/*.parquet'):
        name = file.stem.split('_')[0]
        print(f'Processing {name} ... ')
        table = table_dict[name](Path("/home/data").joinpath(name))
        df = pd.read_parquet(file)
        table.update(df)
        file.unlink()

    data_path.rmdir()

def update_proxy():
    table = quool.FrameTable('/home/data/proxy')
    proxy = table.read()
    print(f'[=] Fetching kaixin proxy source ...')
    kx = quool.KaiXin()()
    print(f'[=] Fetching kuaidaili proxy source ...')
    kdl = quool.KuaiDaili()(para=False)
    print(f'[=] Fetching ip3366 proxy source ...')
    ip3366 = quool.Ip3366()()
    print(f'[=] Fetching ip98 proxy source ...')
    ip98 = quool.Ip98()()
    print(f'[=] Checking availability or proxies ...')
    data = pd.concat([proxy, kx, kdl, ip3366, ip98], ignore_index=True)
    data = data.to_dict(orient='records')
    res = Checker(data)()
    table._write_fragment(res)


if __name__ == "__main__":
    args = parse_args()
    update_data(args)
    update_proxy()
